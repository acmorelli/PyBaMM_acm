"""
A/B comparison: flux_form vs gaussian source methods for CCPM transition sources.
Runs 1C discharge for 2000 s (reaches θ ≈ 0.21 near c_sp1).
Uses Prada2013 parameters, 300 c_p points.
"""
import numpy as np
import pybamm
from pybamm.models.full_battery_models.lithium_ion.dfn_ccpm import DFN_CCPM


def run_one(source_method):
    """Build, solve, return key diagnostics."""
    model = DFN_CCPM(build=True, initial_branch="A", source_method=source_method)
    param = pybamm.ParameterValues("Prada2013")
    experiment = pybamm.Experiment(["Discharge at 1C for 2100 seconds"])
    sim = pybamm.Simulation(
        model,
        parameter_values=param,
        experiment=experiment,
        solver=pybamm.CasadiSolver(mode="safe"),
    )
    sol = sim.solve()

    time = sol["Time [s]"].entries
    V = sol["Voltage [V]"].entries
    theta = sol["X-averaged positive CCPM stoichiometry"].entries
    m_a = sol["X-averaged CCPM Branch A mass"].entries
    m_b = sol["X-averaged CCPM Branch B mass"].entries
    m_c = sol["X-averaged CCPM Branch C mass"].entries
    m_tot = sol["X-averaged CCPM Total mass"].entries
    m_tot_raw = np.mean(sol["CCPM Unmasked Total Mass"].entries, axis=0)
    return {
        "time": time, "V": V, "theta": theta,
        "m_a": m_a, "m_b": m_b, "m_c": m_c,
        "m_tot": m_tot, "m_tot_raw": m_tot_raw,
    }


def print_results(label, r):
    t, V, theta = r["time"], r["V"], r["theta"]
    m_a, m_b, m_c, m_tot = r["m_a"], r["m_b"], r["m_c"], r["m_tot"]
    m_raw = r["m_tot_raw"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Time: {t[0]:.0f}–{t[-1]:.0f} s  ({len(t)} steps)")
    print(f"  Voltage:  {V[0]:.4f} → {V[-1]:.4f} V")
    print(f"  θ_CCPM:   {theta[0]:.6f} → {theta[-1]:.6f}")
    print(f"  MASKED  total mass:  {m_tot[0]:.6f} → {m_tot[-1]:.6f}  drift = {m_tot[-1]-m_tot[0]:+.6e}")
    print(f"  UNMASKED total mass: {m_raw[0]:.6f} → {m_raw[-1]:.6f}  drift = {m_raw[-1]-m_raw[0]:+.6e}")
    print(f"  Branch masses (t=end): A={m_a[-1]:.6f}  B={m_b[-1]:.6f}  C={m_c[-1]:.6f}")


# --- Run both ---
print("Running FLUX_FORM sources …")
r_flux = run_one("flux_form")
print_results("FLUX_FORM (EvaluateAt)", r_flux)

print("\nRunning GAUSSIAN sources …")
r_gauss = run_one("gaussian")
print_results("GAUSSIAN (integral-based)", r_gauss)

# --- Comparison ---
print(f"\n{'='*60}")
print("  COMPARISON")
print(f"{'='*60}")
d_flux = r_flux["m_tot_raw"][-1] - r_flux["m_tot_raw"][0]
d_gauss = r_gauss["m_tot_raw"][-1] - r_gauss["m_tot_raw"][0]
print(f"  Unmasked mass drift (flux_form):  {d_flux:+.6e}")
print(f"  Unmasked mass drift (gaussian):   {d_gauss:+.6e}")
improvement = abs(d_gauss) - abs(d_flux)
if improvement > 0:
    print(f"  Flux-form is BETTER by {improvement:.6e} ({improvement/abs(d_gauss)*100:.1f}%)")
else:
    print(f"  Gaussian is better by {-improvement:.6e}")
print()
