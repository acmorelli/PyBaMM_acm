"""
C/10 discharge until 3.20V — mass diagnostics + interactive Plotly plots.
Reuses the dropdown-style spatial plot from test_dynamics_plots.py.
"""
import numpy as np
import pybamm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Model / solve ──────────────────────────────────────────────────────
model = DFN_CCPM(initial_branch="C", mode="charge")
param = pybamm.ParameterValues("Prada2013")
experiment = pybamm.Experiment(["Charge at C/30 until 3.6V"])

c_max_raw = param["Maximum concentration in positive electrode [mol.m-3]"]
param["Initial concentration in positive electrode [mol.m-3]"] = 0.83 * c_max_raw
# Negative electrode: low θ_n consistent with discharged state
param["Initial concentration in negative electrode [mol.m-3]"] = (
    0.17 * param["Maximum concentration in negative electrode [mol.m-3]"]
)

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
sim.save("C30_300s_modeChg.pkl") #start1207
# ── Extract arrays ─────────────────────────────────────────────────────
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
jtr_a = arr(solution, "CCPM Branch A interfacial current density [A.m-2]")
jtr_b = arr(solution, "CCPM Branch B interfacial current density [A.m-2]")
jtr_c = arr(solution, "CCPM Branch C interfacial current density [A.m-2]")

J_AB = arr(solution, "Branch A to B scalar flux")
J_BA = arr(solution, "Branch B to A scalar flux")
J_BC = arr(solution, "Branch B to C scalar flux")
J_CB = arr(solution, "Branch C to B scalar flux")

# Positive electrode OCP (from CCPM OCP submodel)
ocp_pos = arr(solution, "Positive electrode open-circuit potential [V]")
ocp_pos_avg = arr(solution, "X-averaged positive electrode open-circuit potential [V]")


# ── Console diagnostics ────────────────────────────────────────────────
n = len(time)
print("=== MASS CHECK ===")
print(f"Initial: mA={m_a[0]:.6e}  mB={m_b[0]:.6e}  mC={m_c[0]:.6e}  total={m_tot[0]:.6e}")
print(f"Final:   mA={m_a[-1]:.6e}  mB={m_b[-1]:.6e}  mC={m_c[-1]:.6e}  total={m_tot[-1]:.6e}")
print(f"Total mass drift: {m_tot[-1]-m_tot[0]:+.4e}  ({100*(m_tot[-1]-m_tot[0])/m_tot[0]:+.6f}%)")
print(f"Final time: {time[-1]:.1f}s  ({time[-1]/3600:.2f}h)")
print(f"Final voltage: {voltage[-1]:.4f}V")
print()

print("Time trajectory:")
for frac in np.linspace(0, 1, 11):
    i = min(int(frac * (n - 1)), n - 1)
    jab = np.mean(J_AB[:, i]) if J_AB.ndim > 1 else J_AB[i]
    jbc = np.mean(J_BC[:, i]) if J_BC.ndim > 1 else J_BC[i]
    jba = np.mean(J_BA[:, i]) if J_BA.ndim > 1 else J_BA[i]
    jcb = np.mean(J_CB[:, i]) if J_CB.ndim > 1 else J_CB[i]
    print(
        f"  t={time[i]:7.0f}s  V={voltage[i]:.4f}  theta={theta_ccpm[i]:.6f}"
        f"  mA={m_a[i]:.6f}  mB={m_b[i]:.4e}  mC={m_c[i]:.4e}"
        f"  J_AB={jab:+.3e}  J_BA={jba:+.3e}  J_BC={jbc:+.3e}  J_CB={jcb:+.3e}"
    )
print()

# ── Figure 1: Time-series (masses, theta, voltage, J fluxes) ──────────
fig1 = make_subplots(
    rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
    subplot_titles=("Branch masses", "CCPM stoichiometry", "Terminal voltage", "Positive electrode OCP", "Transition fluxes J_AB, J_BC"),
)

fig1.add_trace(go.Scatter(x=time, y=m_a, mode="lines", name="m_a"), row=1, col=1)
fig1.add_trace(go.Scatter(x=time, y=m_b, mode="lines", name="m_b"), row=1, col=1)
fig1.add_trace(go.Scatter(x=time, y=m_c, mode="lines", name="m_c"), row=1, col=1)
fig1.add_trace(go.Scatter(x=time, y=m_tot, mode="lines", name="m_tot", line=dict(dash="dash")), row=1, col=1)

fig1.add_trace(go.Scatter(x=time, y=theta_ccpm, mode="lines", name="theta_ccpm"), row=2, col=1)

fig1.add_trace(go.Scatter(x=time, y=voltage, mode="lines", name="Voltage [V]"), row=3, col=1)

fig1.add_trace(go.Scatter(x=time, y=ocp_pos_avg, mode="lines", name="OCP [V]"), row=4, col=1)

def _mean_or_1d(a):
    return np.mean(a, axis=0) if a.ndim > 1 else a

fig1.add_trace(go.Scatter(x=time, y=_mean_or_1d(J_AB), mode="lines", name="J_AB"), row=5, col=1)
fig1.add_trace(go.Scatter(x=time, y=_mean_or_1d(J_BA), mode="lines", name="J_BA"), row=5, col=1)
fig1.add_trace(go.Scatter(x=time, y=_mean_or_1d(J_BC), mode="lines", name="J_BC"), row=5, col=1)
fig1.add_trace(go.Scatter(x=time, y=_mean_or_1d(J_CB), mode="lines", name="J_CB"), row=5, col=1)

fig1.update_xaxes(title_text="Time [s]", row=5, col=1)
fig1.update_yaxes(title_text="mass", row=1, col=1)
fig1.update_yaxes(title_text="theta", row=2, col=1)
fig1.update_yaxes(title_text="V", row=3, col=1)
fig1.update_yaxes(title_text="OCP [V]", row=4, col=1)
fig1.update_yaxes(title_text="J [1/s]", row=5, col=1)

fig1.update_layout(height=1200, width=1100, hovermode="x unified",
                   legend_title="Click traces to hide/show",
                   title="C/30 discharge to 3.20V — time series")
fig1.show()

# ── Figure 2: Spatial dropdown plot (g_a, g_b, g_c, R_a, R_b, R_c, j_tr) ─
c_p_max = param["Maximum concentration in positive electrode [mol.m-3]"]
if hasattr(c_p_max, "evaluate"):
    c_p_max = float(c_p_max.evaluate())
else:
    c_p_max = float(c_p_max)

n_cp = model.default_var_pts.get("c_p", 300)
eps_c = 1e-8 * c_p_max
cp_edges = np.linspace(eps_c, c_p_max - eps_c, n_cp + 1)
cp_nodes = 0.5 * (cp_edges[:-1] + cp_edges[1:])
theta_p_nodes = cp_nodes / c_p_max

n_flat = g_a.shape[0] if g_a.ndim == 2 else g_a.size // len(time)
n_x = n_flat // n_cp
n_t = len(time)


def reshape_3d(flat):
    flat = np.squeeze(flat)
    if flat.ndim == 2 and flat.shape[0] == n_cp * n_x:
        return flat.reshape((n_cp, n_x, n_t), order="F")
    elif flat.ndim == 2:
        return flat[:, np.newaxis, :] * np.ones((1, n_x, 1))
    return flat


ga_3d = reshape_3d(g_a)
gb_3d = reshape_3d(g_b)
gc_3d = reshape_3d(g_c)
Ra_3d = reshape_3d(R_a)
Rb_3d = reshape_3d(R_b)
Rc_3d = reshape_3d(R_c)
jtra_3d = reshape_3d(jtr_a)
jtrb_3d = reshape_3d(jtr_b)
jtrc_3d = reshape_3d(jtr_c)

var_catalog = {
    "g_a (Branch A PDF)": ga_3d,
    "g_b (Branch B PDF)": gb_3d,
    "g_c (Branch C PDF)": gc_3d,
    "R_a (Branch A lithiation rate)": Ra_3d,
    "R_b (Branch B lithiation rate)": Rb_3d,
    "R_c (Branch C lithiation rate)": Rc_3d,
    "jtr_a (Branch A current density)": jtra_3d,
    "jtr_b (Branch B current density)": jtrb_3d,
    "jtr_c (Branch C current density)": jtrc_3d,
}

x_idx = [0, n_x // 2, n_x - 1]
x_titles = [f"x = 0 (node {x_idx[0]})", f"x = mid (node {x_idx[1]})", f"x = end (node {x_idx[2]})"]

# Pick 5 time snapshots spread across the simulation
t_idx = [int(f * (n_t - 1)) for f in [0, 0.25, 0.5, 0.75, 1.0]]
t_labels = [f"t={time[i]:.0f}s" for i in t_idx]
color_t = ["blue", "orange", "green", "red", "purple"]
dash_t = ["solid", "dash", "dot", "dashdot", "longdash"]

fig2 = make_subplots(
    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=x_titles,
)

var_names = list(var_catalog.keys())
n_time_snaps = len(t_idx)
n_traces_per_var = 3 * n_time_snaps  # 3 x-locations * n time snapshots

legend_names = ["legend", "legend2", "legend3"]

for v_idx, (vname, data) in enumerate(var_catalog.items()):
    for row, ix in enumerate(x_idx, start=1):
        for it_i, (it, tlab) in enumerate(zip(t_idx, t_labels)):
            fig2.add_trace(
                go.Scatter(
                    x=theta_p_nodes,
                    y=data[:, ix, it],
                    mode="lines",
                    name=tlab,
                    legendgroup=f"row{row}_{tlab}",
                    showlegend=(v_idx == 0),
                    visible=(v_idx == 0),
                    legend=legend_names[row - 1],
                    line=dict(color=color_t[it_i], dash=dash_t[it_i], width=2),
                ),
                row=row, col=1,
            )

buttons = []
for v_idx, vname in enumerate(var_names):
    visibility = []
    for vi in range(len(var_names)):
        visibility.extend([vi == v_idx] * n_traces_per_var)
    showlegends = []
    for vi in range(len(var_names)):
        for row in range(3):
            for it_i in range(n_time_snaps):
                showlegends.append(vi == v_idx)
    buttons.append(dict(
        label=vname,
        method="update",
        args=[
            {"visible": visibility, "showlegend": showlegends},
            {"title": f"CCPM spatial profiles: {vname}"},
        ],
    ))

fig2.update_layout(
    updatemenus=[dict(
        active=0, buttons=buttons,
        x=0.0, y=1.15, xanchor="left", yanchor="top", type="dropdown",
    )],
    height=1000, width=1100,
    title=f"CCPM spatial profiles: {var_names[0]}",
    hovermode="x unified",
    legend=dict(
        title=x_titles[0],
        yanchor="top", y=1.0, xanchor="left", x=1.02,
    ),
    legend2=dict(
        title=x_titles[1],
        yanchor="top", y=0.63, xanchor="left", x=1.02,
    ),
    legend3=dict(
        title=x_titles[2],
        yanchor="top", y=0.30, xanchor="left", x=1.02,
    ),
)

fig2.update_xaxes(title_text="c_p / c_p_max", row=3, col=1)
for r in range(1, 4):
    fig2.update_yaxes(title_text="value", row=r, col=1)

fig2.update_yaxes(exponentformat = 'E')

fig2.show()

# ── Figure 3: x-resolved time series (OCP, J_AB, J_BA, J_BC, J_CB) ──────
# These variables have shape [n_x, n_t] — plot at 3 x-locations.

def reshape_x(flat):
    """Reshape [n_x, n_t] or scalar-over-time to [n_x, n_t]."""
    flat = np.squeeze(flat)
    if flat.ndim == 1:  # scalar (x-averaged), broadcast
        return np.tile(flat, (n_x, 1))
    return flat  # already [n_x, n_t]

ocp_xt = reshape_x(ocp_pos)
JAB_xt = reshape_x(J_AB)
JBA_xt = reshape_x(J_BA)
JBC_xt = reshape_x(J_BC)
JCB_xt = reshape_x(J_CB)

xvar_catalog = {
    "OCP [V]": ocp_xt,
    "J_AB (A→B)": JAB_xt,
    "J_BA (B→A)": JBA_xt,
    "J_BC (B→C)": JBC_xt,
    "J_CB (C→B)": JCB_xt,
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
