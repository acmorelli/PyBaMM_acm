import pickle
import numpy as np
import matplotlib.pyplot as plt
import pybamm

PICKLE_FILE = r"C:\Users\dottavianomo\programming\PyBaMM_acm\my_sim_result_Sbalance.pkl"

with open(PICKLE_FILE, "rb") as f:
    sim = pickle.load(f)

sol = sim.solution
mesh = sim.mesh
parameter_values = sim.parameter_values
model = sol.all_models[0] if isinstance(sol.all_models, list) else sol.all_models
cp_nodes = mesh["CCPM positive particle concentration"].nodes
x_nodes  = mesh["positive electrode"].nodes
t_nodes  = sol.t
t=t_nodes
n_cp = len(cp_nodes)
n_x = len(x_nodes)
n_t = len(t_nodes)

def get_raw_state_key(model, name):
    for k in list(model.rhs.keys()) + list(model.algebraic.keys()):
        if isinstance(k, pybamm.Variable) and k.name == name:
            return k
    raise KeyError(f"Raw state '{name}' not found")

def get_state_array(name):
    key = get_raw_state_key(model, name)
    sl = model.y_slices[key]
    if isinstance(sl, list):
        sl = sl[0]

    Y = sol.y.full() if hasattr(sol.y, "full") else np.asarray(sol.y)
    arr = Y[sl, :]   # shape: (n_cp*n_x, n_t)

    return arr.reshape((n_cp, n_x, n_t), order="F")

ga = get_state_array("Branch A PDF CCPM")
gb = get_state_array("Branch B PDF CCPM")
gc = get_state_array("Branch C PDF CCPM")


t_idx = [0, len(t)//2, len(t)-1]
t_vals = [t[i] for i in t_idx]
t_labels = ["t=0", "t=mid", "t=end"]
# choose x = left, mid, right
x_idx = [0, len(x_nodes)//2, len(x_nodes)-1]
x_titles = ["x = 0", "x = mid", "x = end"]

# choose t = 0, mid, end
t_idx = [0, len(t_nodes)//2, len(t_nodes)-1]
t_labels = ["t = 0", "t = mid", "t = end"]
linestyles = ["-", "--", ":"]
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# positive particle maximum concentration
c_p_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
if hasattr(c_p_max, "evaluate"):
    c_p_max = float(c_p_max.evaluate())
else:
    c_p_max = float(c_p_max)

theta_p_nodes = cp_nodes / c_p_max

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=x_titles,
)

# fixed colors by branch
color_map = {
    "g_a": "blue",
    "g_b": "orange",
    "g_c": "green",
}

# fixed dash by time
dash_map = {
    "t = 0": "solid",
    "t = mid": "dash",
    "t = end": "dot",
}

legend_names = ["legend", "legend2", "legend3"]

for row, (ix, xtitle, legend_name) in enumerate(zip(x_idx, x_titles, legend_names), start=1):
    for it, lab in zip(t_idx, t_labels):
        fig.add_trace(
            go.Scatter(
                x=theta_p_nodes,
                y=ga[:, ix, it],
                mode="lines",
                name=f"g_a, {lab}",
                legend=legend_name,
                line=dict(color=color_map["g_a"], dash=dash_map[lab], width=2),
            ),
            row=row, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=theta_p_nodes,
                y=gb[:, ix, it],
                mode="lines",
                name=f"g_b, {lab}",
                legend=legend_name,
                line=dict(color=color_map["g_b"], dash=dash_map[lab], width=2),
            ),
            row=row, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=theta_p_nodes,
                y=gc[:, ix, it],
                mode="lines",
                name=f"g_c, {lab}",
                legend=legend_name,
                line=dict(color=color_map["g_c"], dash=dash_map[lab], width=2),
            ),
            row=row, col=1
        )

fig.update_yaxes(title_text="density", row=1, col=1)
fig.update_yaxes(title_text="density", row=2, col=1)
fig.update_yaxes(title_text="density", row=3, col=1)
fig.update_xaxes(title_text="c_p / c_{p,max}", row=3, col=1)

fig.update_layout(
    height=1000,
    width=1100,
    title="CCPM branch PDFs",
    hovermode="x unified",

    legend=dict(
        x=1.02, y=1.00,
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="lightgray",
        borderwidth=1,
    ),
    legend2=dict(
        x=1.02, y=0.66,
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="lightgray",
        borderwidth=1,
    ),
    legend3=dict(
        x=1.02, y=0.32,
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="lightgray",
        borderwidth=1,
    ),
)

fig.show()