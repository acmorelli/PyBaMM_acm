import pickle
import numpy as np
import pybamm
import matplotlib.pyplot as plt
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


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

t_eval = np.linspace(0, 40, 51)

print("Solving...")
solution = sim.solve(t_eval=t_eval)
print("Solved.")

cp = sim.mesh["CCPM positive particle concentration"].nodes


# ============================================================
# HELPERS
# ============================================================
def choose_physical_initial_branch(parameter_values):
    c_max = float(parameter_values["Maximum concentration in positive electrode [mol.m-3]"])
    c_init = float(parameter_values["Initial concentration in positive electrode [mol.m-3]"])

    c_sp1 = 0.2113 * c_max
    c_sp2 = 0.7887 * c_max

    print(f"c_init = {c_init:.6e}")
    print(f"c_sp1  = {c_sp1:.6e}")
    print(f"c_sp2  = {c_sp2:.6e}")
    print(f"theta_init = {c_init / c_max:.6f}")

    if c_init < c_sp1:
        return "A"
    elif c_init > c_sp2:
        return "C"
    else:
        raise ValueError(
            "Initial concentration lies inside the spinodal window. "
            "A physical single-branch initialization is not unique there; "
            "you need history/preconditioning."
        )


def arr(sol, name):
    return np.asarray(sol[name].entries).squeeze()


def ensure_cp_time_shape(a, n_cp):
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    if a.shape[0] == n_cp:
        return a
    if a.shape[1] == n_cp:
        return a.T
    raise ValueError(f"Cannot reshape array with shape {a.shape} to (Ncp, Nt)")


def mean_from_pdf(pdf, cp, mass_tol=1e-14):
    mass = np.trapezoid(pdf, cp, axis=0)
    num = np.trapezoid(pdf * cp[:, None], cp, axis=0)

    cbar = np.full_like(mass, np.nan, dtype=float)
    mask = mass > mass_tol
    cbar[mask] = num[mask] / mass[mask]
    return cbar, mass


def sign_report(a):
    a = np.asarray(a).squeeze()
    return f"min={np.min(a):.3e}, max={np.max(a):.3e}"


# ============================================================
# BUILD MODEL WITH PHYSICAL INITIAL BRANCH
# ============================================================
parameter_values = pybamm.ParameterValues(PARAMETER_SET)
initial_branch = choose_physical_initial_branch(parameter_values)

print("Chosen physical CCPM initial branch:", initial_branch)

# For your current parameter set this should be A.
# Starting from A, switching is tested under CHARGE:
# A -> B -> C
model = DFN_CCPM(initial_branch=initial_branch)

parameter_values.update(
    {
        "Current function [A]": CURRENT_A,
        "Lower voltage cut-off [V]": LOWER_CUTOFF_V,
        "Upper voltage cut-off [V]": UPPER_CUTOFF_V,
    }
)

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    solver=SOLVER,
)

t_eval = np.linspace(0, T_END, N_T)

print(f"\nSolving with I = {CURRENT_A:+.3f} A")
solution = sim.solve(t_eval=t_eval)
print("Solved.")


# ============================================================
# EXTRACT DATA
# ============================================================
time = arr(solution, "Time [s]")
voltage = arr(solution, "Terminal voltage [V]")

cp = sim.mesh["CCPM positive particle concentration"].nodes
n_cp = len(cp)

c_max = float(parameter_values["Maximum concentration in positive electrode [mol.m-3]"])
c1_star = 0.0710 * c_max
c_sp1   = 0.2113 * c_max
c_sp2   = 0.7887 * c_max
c2_star = 0.9290 * c_max

gA = ensure_cp_time_shape(arr(solution, "X-averaged Branch A PDF CCPM"), n_cp)
gB = ensure_cp_time_shape(arr(solution, "X-averaged Branch B PDF CCPM"), n_cp)
gC = ensure_cp_time_shape(arr(solution, "X-averaged Branch C PDF CCPM"), n_cp)

mA = arr(solution, "X-averaged CCPM Branch A mass")
mB = arr(solution, "X-averaged CCPM Branch B mass")
mC = arr(solution, "X-averaged CCPM Branch C mass")
mTot = arr(solution, "X-averaged CCPM Total mass")

theta = arr(solution, "X-averaged positive CCPM stoichiometry")
cbar_tot = theta * c_max

# Safe branch means: NaN if branch mass is essentially zero
cbarA, massA_pdf = mean_from_pdf(gA, cp)
cbarB, massB_pdf = mean_from_pdf(gB, cp)
cbarC, massC_pdf = mean_from_pdf(gC, cp)

# Optional rate diagnostics, only if you added these variables
try:
    RA = arr(solution, "X-averaged CCPM Branch A lithiation rate")
    RB = arr(solution, "X-averaged CCPM Branch B lithiation rate")
    RC = arr(solution, "X-averaged CCPM Branch C lithiation rate")
    have_rates = True
except KeyError:
    have_rates = False


# ============================================================
# SUMMARY
# ============================================================
print("\n=== MASS CHECK ===")
print(f"Initial masses:  A={mA[0]:.6e}, B={mB[0]:.6e}, C={mC[0]:.6e}, total={mTot[0]:.6e}")
print(f"Final masses:    A={mA[-1]:.6e}, B={mB[-1]:.6e}, C={mC[-1]:.6e}, total={mTot[-1]:.6e}")
print(f"Total mass drift: {mTot[-1] - mTot[0]:.6e}")

print("\n=== SWITCHING CHECK ===")
print(f"Max B mass = {np.max(mB):.6e}")
print(f"Max C mass = {np.max(mC):.6e}")

if np.max(mB) > 1e-6:
    print("A -> B switching has occurred.")
else:
    print("A -> B switching has NOT occurred yet.")

if np.max(mC) > 1e-6:
    print("B -> C switching has occurred.")
else:
    print("B -> C switching has NOT occurred yet.")

print("\n=== CONCENTRATION CHECK ===")
print(f"cbar_total(0)   = {cbar_tot[0]:.6e}")
print(f"cbar_total(end) = {cbar_tot[-1]:.6e}")
print(f"Delta cbar_total = {cbar_tot[-1] - cbar_tot[0]:.6e}")

print(f"\nc_sp1  = {c_sp1:.6e}")
print(f"c2_star = {c2_star:.6e}")

valid_A = np.isfinite(cbarA)
if np.any(valid_A):
    print(f"Max cbar_A = {np.nanmax(cbarA):.6e}")
    print(f"Final cbar_A = {cbarA[np.where(valid_A)[0][-1]]:.6e}")

print("\n=== VOLTAGE CHECK ===")
print(f"V(0)   = {voltage[0]:.6e}")
print(f"V(end) = {voltage[-1]:.6e}")

if have_rates:
    print("\n=== RATE SIGN CHECK ===")
    print("R_A:", sign_report(RA))
    print("R_B:", sign_report(RB))
    print("R_C:", sign_report(RC))


# ============================================================
# PICK THREE TIMES FOR PDF PLOTS
# ============================================================
i0 = 0
imid = len(time) // 2
iend = len(time) - 1

print("\n=== SNAPSHOT TIMES ===")
print(f"t0   = {time[i0]:.3f} s")
print(f"tmid = {time[imid]:.3f} s")
print(f"tend = {time[iend]:.3f} s")


# ============================================================
# PLOTS
# ============================================================

# 1) Branch masses
plt.figure()
plt.plot(time, mA, label="m_A")
plt.plot(time, mB, label="m_B")
plt.plot(time, mC, label="m_C")
plt.plot(time, mTot, label="m_total")
plt.xlabel("Time [s]")
plt.ylabel("X-averaged branch mass")
plt.title("Branch masses")
plt.legend()
plt.tight_layout()

# 2) Total / branch mean concentrations
plt.figure()
plt.plot(time, cbar_tot, label="total")
plt.plot(time, cbarA, label="A")
plt.plot(time, cbarB, label="B")
plt.plot(time, cbarC, label="C")
plt.axhline(c_sp1, linestyle="--", label="c_sp1")
plt.axhline(c2_star, linestyle="--", label="c2*")
plt.xlabel("Time [s]")
plt.ylabel("Mean concentration")
plt.title("Mean concentrations")
plt.legend()
plt.tight_layout()

# 3) Branch A PDF
plt.figure()
plt.plot(cp, gA[:, i0], label=f"A, t={time[i0]:.1f}s")
plt.plot(cp, gA[:, imid], label=f"A, t={time[imid]:.1f}s")
plt.plot(cp, gA[:, iend], label=f"A, t={time[iend]:.1f}s")
plt.axvline(c1_star, linestyle="--", label="c1*")
plt.axvline(c_sp1, linestyle="--", label="c_sp1")
plt.axvline(c_sp2, linestyle="--", label="c_sp2")
plt.axvline(c2_star, linestyle="--", label="c2*")
plt.xlabel("c_p")
plt.ylabel("X-averaged Branch A PDF")
plt.title("Branch A PDF")
plt.legend()
plt.tight_layout()

# 4) Branch B PDF
plt.figure()
plt.plot(cp, gB[:, i0], label=f"B, t={time[i0]:.1f}s")
plt.plot(cp, gB[:, imid], label=f"B, t={time[imid]:.1f}s")
plt.plot(cp, gB[:, iend], label=f"B, t={time[iend]:.1f}s")
plt.axvline(c1_star, linestyle="--", label="c1*")
plt.axvline(c_sp1, linestyle="--", label="c_sp1")
plt.axvline(c_sp2, linestyle="--", label="c_sp2")
plt.axvline(c2_star, linestyle="--", label="c2*")
plt.xlabel("c_p")
plt.ylabel("X-averaged Branch B PDF")
plt.title("Branch B PDF")
plt.legend()
plt.tight_layout()

# 5) Branch C PDF
plt.figure()
plt.plot(cp, gC[:, i0], label=f"C, t={time[i0]:.1f}s")
plt.plot(cp, gC[:, imid], label=f"C, t={time[imid]:.1f}s")
plt.plot(cp, gC[:, iend], label=f"C, t={time[iend]:.1f}s")
plt.axvline(c1_star, linestyle="--", label="c1*")
plt.axvline(c_sp1, linestyle="--", label="c_sp1")
plt.axvline(c_sp2, linestyle="--", label="c_sp2")
plt.axvline(c2_star, linestyle="--", label="c2*")
plt.xlabel("c_p")
plt.ylabel("X-averaged Branch C PDF")
plt.title("Branch C PDF")
plt.legend()
plt.tight_layout()

# 6) Final combined branch occupancy
plt.figure()
plt.plot(cp, gA[:, iend], label="A final")
plt.plot(cp, gB[:, iend], label="B final")
plt.plot(cp, gC[:, iend], label="C final")
plt.axvline(c1_star, linestyle="--", label="c1*")
plt.axvline(c_sp1, linestyle="--", label="c_sp1")
plt.axvline(c_sp2, linestyle="--", label="c_sp2")
plt.axvline(c2_star, linestyle="--", label="c2*")
plt.xlabel("c_p")
plt.ylabel("X-averaged PDF")
plt.title("Final branch occupancy")
plt.legend()
plt.tight_layout()

# 7) Voltage
plt.figure()
plt.plot(time, voltage)
plt.xlabel("Time [s]")
plt.ylabel("Terminal voltage [V]")
plt.title("Terminal voltage")
plt.tight_layout()

plt.show()