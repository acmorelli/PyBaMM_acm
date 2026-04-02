
import numpy as np
import pybamm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
model = DFN_CCPM(initial_branch="A", mode="discharge")

experiment = pybamm.Experiment([
    "Discharge at C/30 for 300s",    #3.25V
])

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment,
    solver=pybamm.CasadiSolver(mode="safe"),
)

print("solving...")
solution = sim.solve()
# Save the entire simulation (includes model, params, and solution)
#sim.save("1C_2100s_gaussian.pkl")



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

jtr_a = arr(solution, "CCPM Branch A interfacial current density [A.m-2]")
jtr_b = arr(solution, "CCPM Branch B interfacial current density [A.m-2]")
jtr_c = arr(solution, "CCPM Branch C interfacial current density [A.m-2]")

#Mdot_rhs= arr(solution, "Branch A PDF CCPM RHS Integrated on c_p")
#Mdot_fd= np.diff(m_a) / np.diff(time)
Mdot_rhs=0
Mdot_fd=0
# -----------------------------
# Print sanity checks
# -----------------------------

print("=== MASS CHECK ===")
print(f"Initial masses:  A={m_a[0]:.6e}, B={m_b[0]:.6e}, C={m_c[0]:.6e}, total={m_tot[0]:.6e}")
print(f"Final masses:    A={m_a[-1]:.6e}, B={m_b[-1]:.6e}, C={m_c[-1]:.6e}, total={m_tot[-1]:.6e}")
print(f"Total mass drift: {m_tot[-1] - m_tot[0]:.6e}")
print(f"Min/Max total mass over time: {np.min(m_tot):.6e} / {np.max(m_tot):.6e}")
print('PDF RHS integral rhs_a over c_p: ,', Mdot_rhs)
print('derivative mass over time: ',Mdot_fd)
print('R_a: ', R_a)
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

fig = make_subplots(
    rows=4,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=(
        "Branch masses",
        "CCPM stoichiometry",
        "Terminal voltage",
        "Interfacial current density",
    ),
)

# 1) Branch masses
fig.add_trace(go.Scatter(x=time, y=m_a, mode="lines", name="m_a"), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=m_b, mode="lines", name="m_b"), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=m_c, mode="lines", name="m_c"), row=1, col=1)
fig.add_trace(go.Scatter(x=time, y=m_tot, mode="lines", name="m_tot"), row=1, col=1)

# 2) Stoichiometry
fig.add_trace(
    go.Scatter(x=time, y=theta_ccpm, mode="lines", name="theta_ccpm"),
    row=2, col=1
)

# 3) Voltage
fig.add_trace(
    go.Scatter(x=time, y=voltage, mode="lines", name="Terminal voltage [V]"),
    row=3, col=1
)

# 4) Interfacial current density
fig.add_trace(
    go.Scatter(x=time, y=j_ccpm[-1, :], mode="lines", name="j_ccpm"),
    row=4, col=1
)

fig.update_xaxes(title_text="Time [s]", row=4, col=1)
fig.update_yaxes(title_text="X-averaged branch mass", row=1, col=1)
fig.update_yaxes(title_text="X-averaged CCPM stoichiometry", row=2, col=1)
fig.update_yaxes(title_text="Voltage [V]", row=3, col=1)
fig.update_yaxes(title_text="Interfacial current density [A.m-2]", row=4, col=1)

fig.update_layout(
    height=1000,
    width=1100,
    hovermode="x unified",   # one hover box for all visible traces at same x
    legend_title="Click traces to hide/show",
)

fig.show()

# -----------------------------
# Figure 2: Spatial subplots (x=0, x=mid, x=end)
# with dropdown to choose variable (g_a, g_b, g_c, R_a)
# Each subplot has series for t=0, t=mid, t=end
# -----------------------------

# Reconstruct mesh nodes from model defaults and parameter values
c_p_max = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
if hasattr(c_p_max, "evaluate"):
    c_p_max = float(c_p_max.evaluate())
else:
    c_p_max = float(c_p_max)

# CCPM particle concentration grid: 300 uniform FV cells on [eps, c_max - eps]
n_cp = model.default_var_pts.get("c_p", 300)
eps_c = 1e-8 * c_p_max
cp_edges = np.linspace(eps_c, c_p_max - eps_c, n_cp + 1)
cp_nodes = 0.5 * (cp_edges[:-1] + cp_edges[1:])
theta_p_nodes = cp_nodes / c_p_max

# Positive electrode x-grid: default 20 cells
n_x_pts = model.default_var_pts.get("x_p", 20)
# Derive n_x from g_a shape: g_a is (n_cp*n_x, n_t) flattened
n_flat = g_a.shape[0] if g_a.ndim == 2 else g_a.size // len(time)
n_x = n_flat // n_cp
x_nodes = np.linspace(0, 1, n_x)  # normalised

n_t = len(time)

# Reshape (n_cp*n_x, n_t) -> (n_cp, n_x, n_t) with Fortran order
def reshape_3d(flat):
    flat = np.squeeze(flat)
    if flat.ndim == 2 and flat.shape[0] == n_cp * n_x:
        return flat.reshape((n_cp, n_x, n_t), order="F")
    elif flat.ndim == 2:
        # might already be (n_cp, n_t) if x-averaged; broadcast
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

# Variable catalogue: name -> 3d array
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

# Index choices
x_idx = [0, n_x // 2, n_x - 1]
x_titles = [f"x = 0 (node {x_idx[0]})", f"x = mid (node {x_idx[1]})", f"x = end (node {x_idx[2]})"]

t_idx = [0, n_t // 2, n_t - 1]
t_labels = [f"t={time[t_idx[0]]:.1f}s", f"t={time[t_idx[1]]:.1f}s", f"t={time[t_idx[2]]:.1f}s"]

color_t = ["blue", "orange", "green"]
dash_t = ["solid", "dash", "dot"]

fig2 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=x_titles,
)

var_names = list(var_catalog.keys())
n_traces_per_var = 3 * 3  # 3 x-locations * 3 time snapshots

# Add all traces (initially only the first variable is visible)
for v_idx, (vname, data) in enumerate(var_catalog.items()):
    for row, ix in enumerate(x_idx, start=1):
        for it_i, (it, tlab) in enumerate(zip(t_idx, t_labels)):
            fig2.add_trace(
                go.Scatter(
                    x=theta_p_nodes,
                    y=data[:, ix, it],
                    mode="lines",
                    name=f"{tlab}",
                    showlegend=(row == 1 and v_idx == 0),
                    visible=(v_idx == 0),
                    line=dict(color=color_t[it_i], dash=dash_t[it_i], width=2),
                ),
                row=row, col=1,
            )

# Build dropdown buttons
buttons = []
for v_idx, vname in enumerate(var_names):
    visibility = []
    for vi in range(len(var_names)):
        visibility.extend([vi == v_idx] * n_traces_per_var)
    showlegends = []
    for vi in range(len(var_names)):
        for row in range(3):
            for it_i in range(3):
                showlegends.append(row == 0 and vi == v_idx)
    buttons.append(dict(
        label=vname,
        method="update",
        args=[
            {"visible": visibility, "showlegend": showlegends},
            {"title": f"CCPM variable: {vname}"},
        ],
    ))

fig2.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=0.0, y=1.15,
        xanchor="left", yanchor="top",
        type="dropdown",
    )],
    height=1000,
    width=1100,
    title=f"CCPM variable: {var_names[0]}",
    hovermode="x unified",
)

fig2.update_xaxes(title_text="c_p / c_p_max", row=3, col=1)
fig2.update_yaxes(title_text="value", row=1, col=1)
fig2.update_yaxes(title_text="value", row=2, col=1)
fig2.update_yaxes(title_text="value", row=3, col=1)

fig2.show()