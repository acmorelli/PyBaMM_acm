"""
Post-process the saved C/30 discharge to 2.10V simulation (C30_210V_gaussian.pkl).
Reproduces the plots from test_dynamics_plots_c10.py.
"""
import numpy as np
import pybamm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def arr(sol, name):
    return np.asarray(sol[name].entries)


# ── Load saved simulation ──────────────────────────────────────────────
print("Loading C30_210V_gaussian.pkl ...")
sim = pybamm.load("C30_210V_gaussian.pkl")
solution = sim.solution
model = solution.all_models[0]
param = sim.parameter_values
print("Done.\n")

# ── Extract arrays ─────────────────────────────────────────────────────
time = arr(solution, "Time [s]")

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
J_BC = arr(solution, "Branch B to C scalar flux")

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
    print(
        f"  t={time[i]:7.0f}s  V={voltage[i]:.4f}  theta={theta_ccpm[i]:.6f}"
        f"  mA={m_a[i]:.6f}  mB={m_b[i]:.4e}  mC={m_c[i]:.4e}"
        f"  J_AB={jab:+.3e}  J_BC={jbc:+.3e}"
    )
print()

# ── Figure 1: Time-series (masses, theta, voltage, J fluxes) ──────────
n_rows = 4 if J_AB is not None else 3
titles = ["Branch masses", "CCPM stoichiometry", "Terminal voltage"]
if J_AB is not None:
    titles.append("Transition fluxes J_AB, J_BC")

fig1 = make_subplots(
    rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    subplot_titles=titles,
)

fig1.add_trace(go.Scatter(x=time, y=m_a, mode="lines", name="m_a"), row=1, col=1)
fig1.add_trace(go.Scatter(x=time, y=m_b, mode="lines", name="m_b"), row=1, col=1)
fig1.add_trace(go.Scatter(x=time, y=m_c, mode="lines", name="m_c"), row=1, col=1)
fig1.add_trace(go.Scatter(x=time, y=m_tot, mode="lines", name="m_tot", line=dict(dash="dash")), row=1, col=1)

fig1.add_trace(go.Scatter(x=time, y=theta_ccpm, mode="lines", name="theta_ccpm"), row=2, col=1)

fig1.add_trace(go.Scatter(x=time, y=voltage, mode="lines", name="Voltage [V]"), row=3, col=1)

if J_AB is not None and J_BC is not None:
    j_ab_t = np.mean(J_AB, axis=0) if J_AB.ndim > 1 else J_AB
    j_bc_t = np.mean(J_BC, axis=0) if J_BC.ndim > 1 else J_BC
    fig1.add_trace(go.Scatter(x=time, y=j_ab_t, mode="lines", name="J_AB"), row=4, col=1)
    fig1.add_trace(go.Scatter(x=time, y=j_bc_t, mode="lines", name="J_BC"), row=4, col=1)

fig1.update_xaxes(title_text="Time [s]", row=n_rows, col=1)
fig1.update_yaxes(title_text="mass", row=1, col=1)
fig1.update_yaxes(title_text="theta", row=2, col=1)
fig1.update_yaxes(title_text="V", row=3, col=1)
if J_AB is not None:
    fig1.update_yaxes(title_text="J [1/s]", row=4, col=1)

fig1.update_layout(
    height=1000, width=1100, hovermode="x unified",
    legend_title="Click traces to hide/show",
    title="C/30 discharge to 2.10V — time series",
)
fig1.show()

# ── Figure 2: Spatial dropdown plot (g_a, g_b, g_c, + rates/currents if available)
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
    elif flat.ndim == 3 and flat.shape == (n_cp, n_x, n_t):
        return flat
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
x_titles = [
    f"x = 0 (node {x_idx[0]})",
    f"x = mid (node {x_idx[1]})",
    f"x = end (node {x_idx[2]})",
]

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

for v_idx, (vname, data) in enumerate(var_catalog.items()):
    for row, ix in enumerate(x_idx, start=1):
        for it_i, (it, tlab) in enumerate(zip(t_idx, t_labels)):
            fig2.add_trace(
                go.Scatter(
                    x=theta_p_nodes,
                    y=data[:, ix, it],
                    mode="lines",
                    name=tlab,
                    showlegend=(row == 1 and v_idx == 0),
                    visible=(v_idx == 0),
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
                showlegends.append(row == 0 and vi == v_idx)
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
)

fig2.update_xaxes(title_text="c_p / c_p_max", row=3, col=1)
for r in range(1, 4):
    fig2.update_yaxes(title_text="value", row=r, col=1)

fig2.show()
