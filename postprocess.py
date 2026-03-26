import pickle
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# 1) LOAD SAVED OBJECT
# =========================================
PICKLE_FILE = "my_sim_result_Sbalance.pkl"   # <-- change this

with open(PICKLE_FILE, "rb") as f:
    obj = pickle.load(f)

# Works whether you saved the whole Simulation or only the Solution
if hasattr(obj, "solution"):
    sim = obj
    sol = sim.solution
    parameter_values = sim.parameter_values
    mesh = sim.mesh
else:
    sim = None
    sol = obj
    parameter_values = None
    mesh = None

model = sol.all_models[0] if isinstance(sol.all_models, list) else sol.all_models
available = list(model.variables.keys())

print("Termination:", sol.termination)
print("n_t =", len(sol.t))

# =========================================
# 2) VARIABLE NAMES
# =========================================
VARS = {
    "g_a": "Branch A PDF CCPM",
    "g_b": "Branch B PDF CCPM",
    "g_c": "Branch C PDF CCPM",
    "R_a": "CCPM Branch A lithiation rate",
    "R_b": "CCPM Branch B lithiation rate",
    "R_c": "CCPM Branch C lithiation rate",
    "voltage": "Terminal voltage [V]",
}

def search_keys(*needles):
    needles = [n.lower() for n in needles]
    return [k for k in available if all(n in k.lower() for n in needles)]

for key, name in VARS.items():
    if name not in available:
        print(f"\nMissing variable: {name}")
        print("Suggestions:", search_keys(*name.lower().split()[:2]))
        raise KeyError(f"Please fix VARS['{key}']")

sol.update(list(VARS.values()))

# =========================================
# 3) HELPERS
# =========================================
def arr(name):
    return np.squeeze(np.asarray(sol[name].entries))

def cumtrapz_np(y, x):
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out

# =========================================
# 4) GET ARRAYS
# =========================================


g_a_unmasked = arr(VARS["g_a"])
g_b_unmasked = arr(VARS["g_b"])
g_c_unmasked = arr(VARS["g_c"])

R_a = arr(VARS["R_a"])
R_b = arr(VARS["R_b"])
R_c = arr(VARS["R_c"])

voltage = arr(VARS["voltage"])

mask_a_3d = mask_a[:, None, None]
mask_b_3d = mask_b[:, None, None]
mask_c_3d = mask_c[:, None, None]

g_a = g_a_unmasked * mask_a_3d
g_b = g_b_unmasked * mask_b_3d
g_c = g_c_unmasked * mask_c_3d

t = sol.t

print("g_a shape:", g_a.shape)
print("R_a shape:", R_a.shape)

# Expecting shape (n_cp, n_x, n_t)
if g_a.ndim != 3:
    raise ValueError(f"Expected g_a to have shape (n_cp, n_x, n_t), got {g_a.shape}")

n_cp, n_x, n_t = g_a.shape

# =========================================
# 5) MESH NODES
# =========================================
if sim is None:
    raise ValueError("This script needs the saved Simulation object, not only the Solution, because we need mesh + parameters.")

cp_nodes = mesh["CCPM positive particle concentration"][0].nodes
x_nodes = mesh["positive electrode"][0].nodes

if len(cp_nodes) != n_cp:
    raise ValueError(f"cp mesh length {len(cp_nodes)} does not match g_a first dim {n_cp}")
if len(x_nodes) != n_x:
    raise ValueError(f"x mesh length {len(x_nodes)} does not match g_a second dim {n_x}")

# =========================================
# 6) BRANCH GEOMETRY + DELTA WIDTH
# =========================================
c_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]

c1_star = 0.071  * c_max
c_sp1   = 0.2113 * c_max
c_sp2   = 0.7887 * c_max
c2_star = 0.929  * c_max

sigma = 0.007 * c_max   # same as your current source-term code

mask_a = (cp_nodes <= c_sp1).astype(float)
mask_b = ((cp_nodes >= c1_star) & (cp_nodes <= c2_star)).astype(float)
mask_c = (cp_nodes >= c_sp2).astype(float)

def regularized_delta_num(c_target, mask):
    ker = np.exp(-((cp_nodes - c_target) ** 2) / (2 * sigma ** 2)) * mask
    norm = np.trapz(ker, cp_nodes)
    if norm <= 0:
        raise ValueError(f"Delta kernel normalization failed at c_target={c_target}")
    return ker / norm

dA_sp1 = regularized_delta_num(c_sp1, mask_a)
dB_c1  = regularized_delta_num(c1_star, mask_b)
dB_c2  = regularized_delta_num(c2_star, mask_b)
dC_sp2 = regularized_delta_num(c_sp2, mask_c)

# useful target/source kernels too, for reference if needed later
dB_sp1 = regularized_delta_num(c_sp1, mask_b)
dA_c1  = regularized_delta_num(c1_star, mask_a)
dC_c2  = regularized_delta_num(c2_star, mask_c)
dB_sp2 = regularized_delta_num(c_sp2, mask_b)

# =========================================
# 7) RECONSTRUCT TRANSFER FLUXES
# =========================================
# local directional flux densities on (c_p, x, t)
flux_A_to_B = g_a * np.maximum(R_a, 0.0)
flux_B_to_A = g_b * np.maximum(-R_b, 0.0)
flux_B_to_C = g_b * np.maximum(R_b, 0.0)
flux_C_to_B = g_c * np.maximum(-R_c, 0.0)

# scalar transfer magnitudes on (x, t)
J_A_to_B = np.trapz(dA_sp1[:, None, None] * flux_A_to_B, cp_nodes, axis=0)
J_B_to_A = np.trapz(dB_c1[:,  None, None] * flux_B_to_A, cp_nodes, axis=0)
J_B_to_C = np.trapz(dB_c2[:,  None, None] * flux_B_to_C, cp_nodes, axis=0)
J_C_to_B = np.trapz(dC_sp2[:, None, None] * flux_C_to_B, cp_nodes, axis=0)

# x-averaged transfer rates
J_A_to_B_xavg = np.mean(J_A_to_B, axis=0)
J_B_to_A_xavg = np.mean(J_B_to_A, axis=0)
J_B_to_C_xavg = np.mean(J_B_to_C, axis=0)
J_C_to_B_xavg = np.mean(J_C_to_B, axis=0)

# =========================================
# 8) BRANCH MASSES
# =========================================
M_a = np.trapz(mask_a[:, None, None] * g_a, cp_nodes, axis=0)   # (x, t)
M_b = np.trapz(mask_b[:, None, None] * g_b, cp_nodes, axis=0)
M_c = np.trapz(mask_c[:, None, None] * g_c, cp_nodes, axis=0)
M_sum = M_a + M_b + M_c

M_a_xavg = np.mean(M_a, axis=0)
M_b_xavg = np.mean(M_b, axis=0)
M_c_xavg = np.mean(M_c, axis=0)
M_sum_xavg = np.mean(M_sum, axis=0)

# =========================================
# 9) X LOCATIONS TO INSPECT
# =========================================
ix_first = 0
ix_mid   = len(x_nodes) // 2
ix_last  = len(x_nodes) - 1

x_slices = {
    "first positive-electrode node": ix_first,
    "middle positive-electrode node": ix_mid,
    "last positive-electrode node": ix_last,
}

# =========================================
# 10) DIAGNOSTIC PRINTS
# =========================================
print("\n=== MASS DIAGNOSTICS ===")
print("max |M_sum - 1| =", np.max(np.abs(M_sum - 1.0)))
print("M_sum_xavg start/end =", M_sum_xavg[0], M_sum_xavg[-1])

print("\n=== FINAL BRANCH MASSES (x-avg) ===")
print("M_a =", M_a_xavg[-1])
print("M_b =", M_b_xavg[-1])
print("M_c =", M_c_xavg[-1])
print("M_sum =", M_sum_xavg[-1])

# =========================================
# 11) PLOTS
# =========================================

# ---- A) Transfer rates at three x locations ----
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

for ax, (label, ix) in zip(axes, x_slices.items()):
    ax.plot(t, J_A_to_B[ix, :], label=r"$J_{A\to B}$")
    ax.plot(t, J_B_to_A[ix, :], label=r"$J_{B\to A}$")
    ax.plot(t, J_B_to_C[ix, :], label=r"$J_{B\to C}$")
    ax.plot(t, J_C_to_B[ix, :], label=r"$J_{C\to B}$")
    ax.set_title(f"Transfer rates at {label} (x = {x_nodes[ix]:.6g})")
    ax.set_ylabel("rate")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("time [s]")
fig.tight_layout()

# ---- B) x-averaged transfer rates ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t, J_A_to_B_xavg, label=r"$J_{A\to B}$")
ax.plot(t, J_B_to_A_xavg, label=r"$J_{B\to A}$")
ax.plot(t, J_B_to_C_xavg, label=r"$J_{B\to C}$")
ax.plot(t, J_C_to_B_xavg, label=r"$J_{C\to B}$")
ax.set_title("x-averaged transfer rates")
ax.set_xlabel("time [s]")
ax.set_ylabel("rate")
ax.grid(True)
ax.legend()
fig.tight_layout()

# ---- C) Branch masses at three x locations ----
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

for ax, (label, ix) in zip(axes, x_slices.items()):
    ax.plot(t, M_a[ix, :], label=r"$M_a$")
    ax.plot(t, M_b[ix, :], label=r"$M_b$")
    ax.plot(t, M_c[ix, :], label=r"$M_c$")
    ax.plot(t, M_sum[ix, :], label=r"$M_{\mathrm{sum}}$")
    ax.set_title(f"Branch masses at {label} (x = {x_nodes[ix]:.6g})")
    ax.set_ylabel("mass")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("time [s]")
fig.tight_layout()

# ---- D) x-averaged masses + voltage ----
fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

axes[0].plot(t, M_a_xavg, label=r"$M_a$")
axes[0].plot(t, M_b_xavg, label=r"$M_b$")
axes[0].plot(t, M_c_xavg, label=r"$M_c$")
axes[0].plot(t, M_sum_xavg, label=r"$M_{\mathrm{sum}}$")
axes[0].set_title("x-averaged branch masses")
axes[0].set_ylabel("mass")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t, voltage, label="Voltage")
axes[1].set_title("Terminal voltage")
axes[1].set_xlabel("time [s]")
axes[1].set_ylabel("V")
axes[1].grid(True)
axes[1].legend()

fig.tight_layout()

# ---- E) Optional: cumulative transferred mass (x-avg) ----
cum_A_to_B = cumtrapz_np(J_A_to_B_xavg, t)
cum_B_to_A = cumtrapz_np(J_B_to_A_xavg, t)
cum_B_to_C = cumtrapz_np(J_B_to_C_xavg, t)
cum_C_to_B = cumtrapz_np(J_C_to_B_xavg, t)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(t, cum_A_to_B, label=r"$\int J_{A\to B}\,dt$")
ax.plot(t, cum_B_to_A, label=r"$\int J_{B\to A}\,dt$")
ax.plot(t, cum_B_to_C, label=r"$\int J_{B\to C}\,dt$")
ax.plot(t, cum_C_to_B, label=r"$\int J_{C\to B}\,dt$")
ax.set_title("Cumulative x-averaged transferred mass")
ax.set_xlabel("time [s]")
ax.set_ylabel("cumulative transfer")
ax.grid(True)
ax.legend()
fig.tight_layout()

plt.show()