import pybamm
import numpy as np
import pickle
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


def find_state_key(model, name):
    for key in list(model.rhs.keys()) + list(model.algebraic.keys()):
        if isinstance(key, pybamm.Variable) and key.name == name:
            return key
        if isinstance(key, pybamm.Concatenation):
            for child in key.children:
                if isinstance(child, pybamm.Variable) and child.name == name:
                    return child
    raise KeyError(f"State variable '{name}' not found in rhs/algebraic keys")

def extract_state_array(sim, var_name, n_primary, n_secondary):
    sol = sim.solution
    model = sol.all_models[0]
    y = sol.y
    if hasattr(y, "full"):
        y = y.full()
    else:
        y = np.asarray(y)

    var = find_state_key(model, var_name)
    slices = model.y_slices[var]
    if isinstance(slices, slice):
        slices = [slices]

    idx = np.concatenate([np.arange(sl.start, sl.stop) for sl in slices])

    flat = y[idx, :]  # CasADi DM

    n_t = flat.shape[1]
    n_tertiary = flat.shape[0] // (n_primary * n_secondary)

    arr = np.reshape(flat, (n_primary, n_secondary, n_tertiary, n_t))
    return np.squeeze(arr)

cp_nodes = sim.mesh["CCPM positive particle concentration"].nodes
x_nodes = sim.mesh["positive electrode"].nodes

g_a = extract_state_array(sim, "Branch A PDF CCPM", len(cp_nodes), len(x_nodes))
g_b = extract_state_array(sim, "Branch B PDF CCPM", len(cp_nodes), len(x_nodes))
g_c = extract_state_array(sim, "Branch C PDF CCPM", len(cp_nodes), len(x_nodes))

print("g_a shape:", g_a.shape)
print("g_b shape:", g_b.shape)
print("g_c shape:", g_c.shape)

import matplotlib.pyplot as plt

# time indices
it0 = 0
itmid = len(sol.t) // 2
itend = len(sol.t) - 1
time_ids = [it0, itmid, itend]
time_labels = [f"t={sol.t[i]:.1f} s" for i in time_ids]

# x indices
ix0 = 0
ixmid = len(x_nodes) // 2
ixlast = len(x_nodes) - 1
x_cases = [
    ("x = first node", ix0),
    ("x = middle node", ixmid),
    ("x = last node", ixlast),
]

# optional normalization for x-axis
c_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]

fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

for ax, (x_label, ix) in zip(axes, x_cases):
    for it, tlabel in zip(time_ids, time_labels):
        ax.plot(cp_nodes / c_max, g_a[:, ix, it], label=f"g_a, {tlabel}")
        ax.plot(cp_nodes / c_max, g_b[:, ix, it], label=f"g_b, {tlabel}")
        ax.plot(cp_nodes / c_max, g_c[:, ix, it], label=f"g_c, {tlabel}")

    ax.set_title(f"Densities at {x_label} (x={x_nodes[ix]:.4e} m)")
    ax.set_ylabel("density")
    ax.grid(True)
    ax.legend(fontsize=8, ncol=3)

axes[-1].set_xlabel(r"$c_p / c_{max}$")
plt.tight_layout()
plt.show()