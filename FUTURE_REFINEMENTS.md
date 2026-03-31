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
