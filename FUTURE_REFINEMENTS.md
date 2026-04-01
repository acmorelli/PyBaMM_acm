# Future Refinements / Optimizations

## Exact initial distribution from dataset

Currently `set_initial_conditions` builds a Gamma(2,λ) × damping analytical
distribution and uses a two-pass λ correction so the discrete mean matches
`c_init`.  This is parameter-set agnostic but still approximate (~0.3 % error
on θ₀).

For a truly exact initial state we could:

1. Run a very short "calibration" solve (or offline script) with the target
   parameter set to obtain the steady-state PDF on the discrete mesh.
2. Store that 300-point vector (one per branch) alongside the parameter file
   (e.g. as a `.npy` or directly inside the parameter dict).
3. At model build time, pass the pre-computed PDF array as a parameter to
   `DFN_CCPM` / `CCPMPositiveParticle` and use it directly in
   `set_initial_conditions` instead of the analytical Gamma construction.

This would eliminate any residual mean error and allow arbitrarily shaped
initial distributions (e.g. partially transitioned A+B states from a previous
simulation checkpoint).

## Numerical noise from bidirectional transition fluxes

After introducing charge capability (bidirectional C→B, B→A fluxes alongside
the existing A→B, B→C), we observed increased numerical noise in the solution.
The root cause is that all four transition fluxes are always evaluated and
sigmoid-gated, even when only one direction is physically active. The sigmoid
Heaviside `pybamm.sigmoid(0, J, 1e4)` suppresses the wrong-direction flux but
does not zero it exactly — residual O(1e-15) leakage accumulates over long
solves and pollutes the mass balance.

**Proposed fix: explicit charge/discharge mode separation.**

Rather than relying on the sigmoid to gate both directions simultaneously, the
model could accept a `mode` flag (`"discharge"` / `"charge"` / `"auto"`):

- **`"discharge"`**: Only A→B and B→C sources are built; C→B and B→A are
  identically zero (not just small — never constructed).
- **`"charge"`**: Only C→B and B→A sources are built; A→B and B→C are zero.
- **`"auto"` (default)**: Current behaviour — all four fluxes active with
  sigmoid gating, accepting the small noise penalty for generality.

For multi-step experiments (e.g. discharge then charge), the mode could switch
between experiment steps, or the `"auto"` mode could be kept with a tighter
sigmoid steepness (k > 1e4) at the cost of solver stiffness.

This would:
1. Eliminate residual leakage in single-direction experiments.
2. Reduce the number of symbolic expressions the solver tracks (4 → 2 fluxes),
   potentially improving solve speed.
3. Keep the general bidirectional mode available for cycling or rest periods
   where the current direction may change.
