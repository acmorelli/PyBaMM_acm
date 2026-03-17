import numpy as np
import matplotlib.pyplot as plt
import pybamm

from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


# ============================================================
# SETTINGS
# ============================================================
# PyBaMM standard convention is usually:
#   + current = discharge
#   - current = charge
#
# For transition testing, 50 s is often too short.
# Increase T_END or |CURRENT_MAG| if you do not reach a transition region.

CURRENT_MAG = 3.0   # [A]
T_END = 2000.0      # [s]
N_T = 101

SOLVER = pybamm.CasadiSolver(mode="safe")


# ============================================================
# HELPERS
# ============================================================
def arr(sol, name):
    return np.asarray(sol[name].entries).squeeze()


def sign_status(a):
    a = np.asarray(a).squeeze()
    return f"min={np.min(a):.3e}, max={np.max(a):.3e}"


def mean_from_pdf(pdf, cp):
    """
    pdf shape: (Ncp, Nt)
    returns cbar(t)
    """
    mass = np.trapezoid(pdf, cp, axis=0)
    num = np.trapezoid(pdf * cp[:, None], cp, axis=0)
    return num / np.maximum(mass, 1e-16)


def window_mass(pdf, cp, c0, halfwidth):
    mask = np.abs(cp[:, None] - c0) <= halfwidth
    masked = np.where(mask, pdf, 0.0)
    return np.trapezoid(masked, cp, axis=0)


def run_case(current_A, label):
    model = DFN_CCPM()

    param = model.default_parameter_values.copy()
    param.update({"Current function [A]": current_A})

    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        solver=SOLVER,
    )

    t_eval = np.linspace(0, T_END, N_T)

    print(f"\n=== SOLVING CASE: {label} (I = {current_A:+.3f} A) ===")
    sol = sim.solve(t_eval=t_eval)
    print("Solved.")

    cp = sim.mesh["CCPM positive particle concentration"].nodes
    time = arr(sol, "Time [s]")

    # PDFs
    gA = arr(sol, "X-averaged Branch A PDF CCPM")
    gB = arr(sol, "X-averaged Branch B PDF CCPM")
    gC = arr(sol, "X-averaged Branch C PDF CCPM")

    # Ensure shape is (Ncp, Nt)
    if gA.shape[0] != len(cp):
        gA = gA.T
    if gB.shape[0] != len(cp):
        gB = gB.T
    if gC.shape[0] != len(cp):
        gC = gC.T

    # Masses
    mA = arr(sol, "X-averaged CCPM Branch A mass")
    mB = arr(sol, "X-averaged CCPM Branch B mass")
    mC = arr(sol, "X-averaged CCPM Branch C mass")
    mTot = arr(sol, "X-averaged CCPM Total mass")

    # Stoichiometry / total mean concentration
    theta = arr(sol, "X-averaged positive CCPM stoichiometry")
    c_max = float(param.evaluate(model.param.p.prim.c_max))
    cbar_tot = theta * c_max

    # Branch mean concentrations
    cbarA = mean_from_pdf(gA, cp)
    cbarB = mean_from_pdf(gB, cp)
    cbarC = mean_from_pdf(gC, cp)

    # Rates
    RA = arr(sol, "X-averaged CCPM Branch A lithiation rate")
    RB = arr(sol, "X-averaged CCPM Branch B lithiation rate")
    RC = arr(sol, "X-averaged CCPM Branch C lithiation rate")

    # Voltage
    V = arr(sol, "Terminal voltage [V]")

    # Transition concentrations
    c1_star = 0.0710 * c_max
    c_sp1   = 0.2113 * c_max
    c_sp2   = 0.7887 * c_max
    c2_star = 0.9290 * c_max

    dcp = cp[1] - cp[0]
    halfwidth = max(2 * dcp, 0.02 * c_max)

    # Mass in windows around transitions
    wA_csp1 = window_mass(gA, cp, c_sp1, halfwidth)
    wB_c1s  = window_mass(gB, cp, c1_star, halfwidth)
    wB_c2s  = window_mass(gB, cp, c2_star, halfwidth)
    wC_csp2 = window_mass(gC, cp, c_sp2, halfwidth)

    out = {
        "label": label,
        "current_A": current_A,
        "time": time,
        "cp": cp,
        "c_max": c_max,
        "gA": gA,
        "gB": gB,
        "gC": gC,
        "mA": mA,
        "mB": mB,
        "mC": mC,
        "mTot": mTot,
        "theta": theta,
        "cbar_tot": cbar_tot,
        "cbarA": cbarA,
        "cbarB": cbarB,
        "cbarC": cbarC,
        "RA": RA,
        "RB": RB,
        "RC": RC,
        "V": V,
        "c1_star": c1_star,
        "c_sp1": c_sp1,
        "c_sp2": c_sp2,
        "c2_star": c2_star,
        "wA_csp1": wA_csp1,
        "wB_c1s": wB_c1s,
        "wB_c2s": wB_c2s,
        "wC_csp2": wC_csp2,
    }
    return out


def summarize(data):
    label = data["label"]
    I = data["current_A"]

    mA = data["mA"]
    mB = data["mB"]
    mC = data["mC"]
    mTot = data["mTot"]

    cbar_tot = data["cbar_tot"]
    cbarA = data["cbarA"]
    cbarB = data["cbarB"]
    cbarC = data["cbarC"]

    RA = data["RA"]
    RB = data["RB"]
    RC = data["RC"]

    wA_csp1 = data["wA_csp1"]
    wB_c1s  = data["wB_c1s"]
    wB_c2s  = data["wB_c2s"]
    wC_csp2 = data["wC_csp2"]

    tol = 1e-6

    print(f"\n=== SUMMARY: {label} ===")

    print("\nMass transfer / conservation")
    print(f"Initial masses: A={mA[0]:.6e}, B={mB[0]:.6e}, C={mC[0]:.6e}, total={mTot[0]:.6e}")
    print(f"Final masses:   A={mA[-1]:.6e}, B={mB[-1]:.6e}, C={mC[-1]:.6e}, total={mTot[-1]:.6e}")
    print(f"Total drift:    {mTot[-1] - mTot[0]:.6e}")
    print(f"Max B mass:     {np.max(mB):.6e}")
    print(f"Max A mass:     {np.max(mA):.6e}")

    print("\nMixed-phase occupancy")
    if np.max(mB) > tol:
        print("Branch B becomes occupied.")
    else:
        print("Branch B stays essentially empty.")

    print("\nBehavior near transition concentrations")
    print(f"max mass of A near c_sp1 = {np.max(wA_csp1):.6e}")
    print(f"max mass of B near c1*   = {np.max(wB_c1s):.6e}")
    print(f"max mass of B near c2*   = {np.max(wB_c2s):.6e}")
    print(f"max mass of C near c_sp2 = {np.max(wC_csp2):.6e}")

    print("\nBranch mean concentrations")
    print(f"Delta cbar_total = {cbar_tot[-1] - cbar_tot[0]:.6e}")
    print(f"Delta cbar_A     = {cbarA[-1] - cbarA[0]:.6e}")
    print(f"Delta cbar_B     = {cbarB[-1] - cbarB[0]:.6e}")
    print(f"Delta cbar_C     = {cbarC[-1] - cbarC[0]:.6e}")

    print("\nRate sign check")
    print(f"R_A: {sign_status(RA)}")
    print(f"R_B: {sign_status(RB)}")
    print(f"R_C: {sign_status(RC)}")

    print("\nCharge/discharge sign convention")
    if I < 0:
        print("This is a CHARGE case (negative current).")
        print("Expected for positive electrode: lithium insertion -> concentration should tend to move right / increase.")
    else:
        print("This is a DISCHARGE case (positive current).")
        print("Expected for positive electrode: lithium extraction -> concentration should tend to move left / decrease.")

    delta = cbar_tot[-1] - cbar_tot[0]
    if delta > 0:
        print("Observed: total cathode mean concentration increased.")
    elif delta < 0:
        print("Observed: total cathode mean concentration decreased.")
    else:
        print("Observed: total cathode mean concentration stayed constant.")

    if I < 0 and delta > 0:
        print("Sign convention looks consistent for charge.")
    elif I > 0 and delta < 0:
        print("Sign convention looks consistent for discharge.")
    else:
        print("Sign convention looks suspicious or transitions are dominating / not reached yet.")


def plot_case(data):
    time = data["time"]
    cp = data["cp"]

    gA = data["gA"]
    gB = data["gB"]
    gC = data["gC"]

    mA = data["mA"]
    mB = data["mB"]
    mC = data["mC"]
    mTot = data["mTot"]

    cbar_tot = data["cbar_tot"]
    cbarA = data["cbarA"]
    cbarB = data["cbarB"]
    cbarC = data["cbarC"]

    c1_star = data["c1_star"]
    c_sp1   = data["c_sp1"]
    c_sp2   = data["c_sp2"]
    c2_star = data["c2_star"]

    label = data["label"]

    i0 = 0
    imid = len(time) // 2
    iend = len(time) - 1

    # 1) branch masses
    plt.figure()
    plt.plot(time, mA, label="m_A")
    plt.plot(time, mB, label="m_B")
    plt.plot(time, mC, label="m_C")
    plt.plot(time, mTot, label="m_total")
    plt.xlabel("Time [s]")
    plt.ylabel("X-averaged branch mass")
    plt.title(f"{label}: branch masses")
    plt.legend()
    plt.tight_layout()

    # 2) mean concentrations
    plt.figure()
    plt.plot(time, cbar_tot, label="total")
    plt.plot(time, cbarA, label="A")
    plt.plot(time, cbarB, label="B")
    plt.plot(time, cbarC, label="C")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean concentration")
    plt.title(f"{label}: mean concentrations")
    plt.legend()
    plt.tight_layout()

    # 3) A PDF
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
    plt.title(f"{label}: Branch A PDF")
    plt.legend()
    plt.tight_layout()

    # 4) B PDF
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
    plt.title(f"{label}: Branch B PDF")
    plt.legend()
    plt.tight_layout()

    # 5) C PDF
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
    plt.title(f"{label}: Branch C PDF")
    plt.legend()
    plt.tight_layout()

    # 6) final branch occupancy
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
    plt.title(f"{label}: final branch occupancy")
    plt.legend()
    plt.tight_layout()


# ============================================================
# MAIN
# ============================================================
# +I = discharge, -I = charge
discharge = run_case(+CURRENT_MAG, "DISCHARGE")
#charge = run_case(-CURRENT_MAG, "CHARGE")

summarize(discharge)
#summarize(charge)

plot_case(discharge)
#plot_case(charge)

plt.show()