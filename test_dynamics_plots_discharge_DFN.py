"""
C/10 discharge until 3.20V — mass diagnostics + interactive Plotly plots.
Reuses the dropdown-style spatial plot from test_dynamics_plots.py.
"""
import numpy as np
import pybamm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Model / solve ──────────────────────────────────────────────────────
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues("Prada2013")
experiment = pybamm.Experiment(["Discharge at C/30 until 2.0V"])

sim = pybamm.Simulation(
    model,
    parameter_values=param,
    experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)

print("solving...")
pybamm.set_logging_level("INFO") 
try:
    solution = sim.solve() #1911
except Exception as e:
    print(f"Solver failed: {e}")
    import traceback; traceback.print_exc()
print("done.\n")
sim.save("C30_2V_modeDch_DFN.pkl") #start1207
# ── Extract arrays ─────────────────────────────────────────────────────
time = solution["Time [s]"].entries

# Positive electrode OCP (from CCPM OCP submodel)
ocp_pos = arr(solution, "Positive electrode open-circuit potential [V]")
ocp_pos_avg = arr(solution, "X-averaged positive electrode open-circuit potential [V]")
voltage = arr(solution, "Voltage [V]")

# Negative electrode OCP
ocp_neg = arr(solution, "Negative electrode open-circuit potential [V]")
ocp_neg_avg = arr(solution, "X-averaged negative electrode open-circuit potential [V]")

# Cathode stoichiometry
sto_pos = arr(solution, "Positive particle surface stoichiometry")
sto_pos_avg = arr(solution, "X-averaged positive particle surface stoichiometry")

# ── Console diagnostics ────────────────────────────────────────────────
n = len(time)

# Derive n_x from the spatial OCP shape
ocp_pos_sq = np.squeeze(ocp_pos)
n_x = ocp_pos_sq.shape[0] if ocp_pos_sq.ndim == 2 else 1


# ── Figure 1: Time-series (masses, theta, voltage, J fluxes) ──────────
fig1 = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=("Terminal voltage", "Positive electrode OCP", "Cathode stoichiometry (x-avg)"),
)

fig1.add_trace(go.Scatter(x=time, y=voltage, mode="lines", name="Voltage [V]"), row=1, col=1)

fig1.add_trace(go.Scatter(x=time, y=ocp_pos_avg, mode="lines", name="OCP pos [V]"), row=2, col=1)
fig1.add_trace(go.Scatter(x=time, y=ocp_neg_avg, mode="lines", name="OCP neg [V]"), row=2, col=1)

fig1.add_trace(go.Scatter(x=time, y=sto_pos_avg, mode="lines", name="θ_p (x-avg)"), row=3, col=1)

fig1.update_xaxes(title_text="Time [s]", row=3, col=1)
fig1.update_yaxes(title_text="V", row=1, col=1)
fig1.update_yaxes(title_text="OCP [V]", row=2, col=1)
fig1.update_yaxes(title_text="θ_p", row=3, col=1)
fig1.update_layout(height=800, width=1100, hovermode="x unified",
                   title="C/30 discharge to 2.0V — DFN time series")
fig1.show()

# ── Figure 2: Spatial dropdown plot (g_a, g_b, g_c, R_a, R_b, R_c, j_tr) ─

# ── Figure 3: x-resolved time series (OCP, J_AB, J_BA, J_BC, J_CB) ──────
# These variables have shape [n_x, n_t] — plot at 3 x-locations.

def reshape_x(flat):
    """Reshape [n_x, n_t] or scalar-over-time to [n_x, n_t]."""
    flat = np.squeeze(flat)
    if flat.ndim == 1:  # scalar (x-averaged), broadcast
        return np.tile(flat, (n_x, 1))
    return flat  # already [n_x, n_t]

ocp_pos_xt = reshape_x(ocp_pos)
ocp_neg_xt = reshape_x(ocp_neg)
sto_pos_xt = reshape_x(sto_pos)
x_idx = [0, n_x // 2, n_x - 1]
x_titles = [f"x = 0 (node {x_idx[0]})", f"x = mid (node {x_idx[1]})", f"x = end (node {x_idx[2]})"]

xvar_catalog = {
    "OCP pos [V]": ocp_pos_xt,
    "OCP neg [V]": ocp_neg_xt,
    "θ_p (stoich)": sto_pos_xt,
}

colors_x = ["blue", "orange", "green"]
fig3 = go.Figure()

xvar_names = list(xvar_catalog.keys())
n_x_locs = len(x_idx)
n_traces_per_xvar = n_x_locs

for vi, (vname, data) in enumerate(xvar_catalog.items()):
    for xi_i, ix in enumerate(x_idx):
        fig3.add_trace(go.Scatter(
            x=time, y=data[ix, :],
            mode="lines",
            name=x_titles[xi_i],
            visible=(vi == 0),
            showlegend=(vi == 0),
            line=dict(color=colors_x[xi_i], width=2),
        ))

buttons3 = []
for vi, vname in enumerate(xvar_names):
    vis = []
    for vj in range(len(xvar_names)):
        vis.extend([vj == vi] * n_traces_per_xvar)
    buttons3.append(dict(
        label=vname,
        method="update",
        args=[{"visible": vis, "showlegend": [vj == vi for vj in range(len(xvar_names)) for _ in range(n_traces_per_xvar)]},
              {"title": f"x-resolved time series: {vname}", "yaxis.title.text": vname}],
    ))

fig3.update_layout(
    updatemenus=[dict(
        active=0, buttons=buttons3,
        x=0.0, y=1.12, xanchor="left", yanchor="top", type="dropdown",
    )],
    height=500, width=1100,
    title=f"x-resolved time series: {xvar_names[0]}",
    xaxis_title="Time [s]",
    yaxis_title=xvar_names[0],
    hovermode="x unified",
)
fig3.show()
