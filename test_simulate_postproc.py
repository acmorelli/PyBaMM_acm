import pickle
import numpy as np
import matplotlib.pyplot as plt
import pybamm

# ============================================================
# USER SETTINGS
# ============================================================
BUNDLE_FILE = "ccpm_run_bundle.pkl"

# only needed if c_max was NOT saved in the bundle
FALLBACK_PARAMETER_SET = "Prada2013"


# ============================================================
# HELPERS
# ============================================================
def arr(solution, name):
    return np.asarray(solution[name].entries).squeeze()


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


def try_get(solution, name):
    try:
        return arr(solution, name), True
    except KeyError:
        return None, False


# ============================================================
# LOAD SAVED RUN
# ============================================================
with open(BUNDLE_FILE, "rb") as f:
    bundle = pickle.load(f)

solution = bundle["solution"]
cp = np.asarray(bundle["cp"]).squeeze()

parameter_set_name = bundle.get("parameter_set", FALLBACK_PARAMETER_SET)
initial_branch = bundle.get("initial_branch", "unknown")

print("Loaded bundle:", BUNDLE_FILE)
print("Parameter set:", parameter_set_name)
print("Initial branch:", initial_branch)

n_cp = len(cp)

# ------------------------------------------------------------
# c_max and transition concentrations
# ------------------------------------------------------------
if "c_max" in bundle:
    c_max = float(bundle["c_max"])
else:
    parameter_values = pybamm.ParameterValues(parameter_set_name)
    c_max = float(parameter_values["Maximum concentration in positive electrode [mol.m-3]"])

c1_star = 0.0710 * c_max
c_sp1   = 0.2113 * c_max
c_sp2   = 0.7887 * c_max
c2_star = 0.9290 * c_max


# ============================================================
# EXTRACT DATA
# ============================================================
time = arr(solution, "Time [s]")
voltage = arr(solution, "Terminal voltage [V]")

# PDFs: optional, depending on what you saved as output variables
gA = gB = gC = None
have_gA = have_gB = have_gC = False

tmp, have_gA = try_get(solution, "X-averaged Branch A PDF CCPM")
if have_gA:
    gA = ensure_cp_time_shape(tmp, n_cp)

tmp, have_gB = try_get(solution, "X-averaged Branch B PDF CCPM")
if have_gB:
    gB = ensure_cp_time_shape(tmp, n_cp)

tmp, have_gC = try_get(solution, "X-averaged Branch C PDF CCPM")
if have_gC:
    gC = ensure_cp_time_shape(tmp, n_cp)

# masses
mA = arr(solution, "X-averaged CCPM Branch A mass")
mB = arr(solution, "X-averaged CCPM Branch B mass")
mC = arr(solution, "X-averaged CCPM Branch C mass")
mTot = arr(solution, "X-averaged CCPM Total mass")

# stoichiometry / total mean concentration
theta = arr(solution, "X-averaged positive CCPM stoichiometry")
cbar_tot = theta * c_max

# safe branch means
if have_gA:
    cbarA, massA_pdf = mean_from_pdf(gA, cp)
else:
    cbarA = np.full_like(time, np.nan, dtype=float)

if have_gB:
    cbarB, massB_pdf = mean_from_pdf(gB, cp)
else:
    cbarB = np.full_like(time, np.nan, dtype=float)

if have_gC:
    cbarC, massC_pdf = mean_from_pdf(gC, cp)
else:
    cbarC = np.full_like(time, np.nan, dtype=float)

# optional rate diagnostics
RA, have_RA = try_get(solution, "X-averaged CCPM Branch A lithiation rate")
RB, have_RB = try_get(solution, "X-averaged CCPM Branch B lithiation rate")
RC, have_RC = try_get(solution, "X-averaged CCPM Branch C lithiation rate")
have_rates = have_RA and have_RB and have_RC


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
print(f"cbar_total(0)    = {cbar_tot[0]:.6e}")
print(f"cbar_total(end)  = {cbar_tot[-1]:.6e}")
print(f"Delta cbar_total = {cbar_tot[-1] - cbar_tot[0]:.6e}")

print(f"\nc_sp1   = {c_sp1:.6e}")
print(f"c2_star = {c2_star:.6e}")

if np.any(np.isfinite(cbarA)):
    print(f"Max cbar_A   = {np.nanmax(cbarA):.6e}")
    print(f"Final cbar_A = {cbarA[np.where(np.isfinite(cbarA))[0][-1]]:.6e}")
else:
    print("cbar_A unavailable (X-averaged Branch A PDF CCPM was not saved).")

if np.any(np.isfinite(cbarB)):
    print(f"Max cbar_B   = {np.nanmax(cbarB):.6e}")
    print(f"Final cbar_B = {cbarB[np.where(np.isfinite(cbarB))[0][-1]]:.6e}")
else:
    print("cbar_B unavailable (X-averaged Branch B PDF CCPM was not saved).")

if np.any(np.isfinite(cbarC)):
    print(f"Max cbar_C   = {np.nanmax(cbarC):.6e}")
    print(f"Final cbar_C = {cbarC[np.where(np.isfinite(cbarC))[0][-1]]:.6e}")
else:
    print("cbar_C unavailable (X-averaged Branch C PDF CCPM was not saved).")

print("\n=== VOLTAGE CHECK ===")
print(f"V(0)   = {voltage[0]:.6e}")
print(f"V(end) = {voltage[-1]:.6e}")

if have_rates:
    print("\n=== RATE SIGN CHECK ===")
    print("R_A:", sign_report(RA))
    print("R_B:", sign_report(RB))
    print("R_C:", sign_report(RC))
else:
    print("\nRate diagnostics unavailable (x-averaged CCPM branch rates were not saved).")


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
if np.any(np.isfinite(cbarA)):
    plt.plot(time, cbarA, label="A")
if np.any(np.isfinite(cbarB)):
    plt.plot(time, cbarB, label="B")
if np.any(np.isfinite(cbarC)):
    plt.plot(time, cbarC, label="C")
plt.axhline(c_sp1, linestyle="--", label="c_sp1")
plt.axhline(c2_star, linestyle="--", label="c2*")
plt.xlabel("Time [s]")
plt.ylabel("Mean concentration")
plt.title("Mean concentrations")
plt.legend()
plt.tight_layout()

# 3) Branch A PDF
if have_gA:
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
if have_gB:
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
if have_gC:
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
if have_gA and have_gB and have_gC:
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