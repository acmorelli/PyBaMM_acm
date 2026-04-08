# CCPM Snap-Mesh Implementation Report

## `ccpm_snap_mesh_Ra_smooth` Worktree — Detailed Description

**Author**: Auto-generated from commit history, repo memory, and source analysis  
**Date**: 2026-04-08  
**Branch**: `ccpm_snap_mesh_Ra_smooth` (26+ commits, 2026-03-14 → 2026-04-08)  
**Base**: PyBaMM `main` (v25.12.2+)

---

## 1. What This Project Is

This worktree implements the **Clarke 2026 Continuous Composition Particle Model (CCPM)** for LFP (lithium iron phosphate) batteries inside PyBaMM. Unlike the standard Doyle–Fuller–Newman (DFN) model, which tracks radial diffusion inside each particle using Fick's law, the CCPM represents the cathode as a **probability density function (PDF)** over particle lithium concentration. Three PDFs — one per thermodynamic branch — evolve via advection PDEs driven by electrochemical reaction rates.

The key physics captured is **LFP phase-separation hysteresis**: particles transition between a Li-poor α phase (Branch A), a mixed two-phase region (Branch B), and a Li-rich β phase (Branch C). The model tracks a continuous distribution of particle compositions rather than assuming a single representative particle.

---

## 2. Architecture Overview

### 2.1 Inheritance Strategy

The implementation extends PyBaMM's existing `DFN` model rather than building from scratch. Only three submodels are replaced:

| Submodel Slot | Standard DFN | CCPM Replacement |
|---|---|---|
| Positive particle | `FickianDiffusion` | `CCPMPositiveParticle` |
| Positive OCP | `SingleOpenCircuitPotential` | `CCPMOpenCircuitPotential` |
| Positive kinetics | `ButlerVolmer` | `CCPMPositiveInterface` |

The negative electrode (graphite), electrolyte transport, thermal model, and current collectors remain unmodified. This lets the CCPM cathode couple naturally into the DFN's macroscopic conservation equations ($x$-direction ionic/electronic transport).

**File**: `src/pybamm/models/full_battery_models/lithium_ion/dfn_ccpm.py` (~130 lines)

#### Constructor Signature

```python
DFN_CCPM(
    options=None,
    initial_branch="A",
    mode="discharge",
    parameter_values=None,  # required — used to extract R_p and c_max as floats
    npts=300,               # number of FV cells on concentration grid
    name="DFN_CCPM",
    build=True,
)
```

`parameter_values` **must** be supplied before model construction because the particle radius $R_p$ and maximum concentration $c_{\max}$ are needed at build time to compute the size-dependent Ω and snap branch boundaries to the FV mesh. Internally, DFN_CCPM extracts:

```python
self._ccpm_R_p = float(parameter_values["Positive particle radius [m]"])
self._ccpm_c_max = float(parameter_values["Maximum concentration in positive electrode [mol.m-3]"])
```

These floats are passed to `CCPMPositiveParticle` and `CCPMPositiveInterface` submodels. The `npts` parameter flows to `default_var_pts["c_p"]` and to all CCPM submodel constructors.

### 2.2 Single-Domain Architecture (vs. Three-Subdomain in PyBaMM_acm)

This worktree uses a **single custom domain** for all three PDF branches:

```
Domain: "CCPM positive particle concentration"
Spatial variable: c_p
Range: [eps_c, c_max − eps_c]  =  [2.28e-4, 22805.99977] mol/m³
Points: npts (default 300, configurable via DFN_CCPM constructor)
Spacing: Δc ≈ c_max / npts
```

Branches are distinguished by **boolean masks** applied to the shared grid. The mask cell indices depend on `npts` and the particle radius $R_p$ (via the size-dependent Ω). For example, with npts=300 and the Prada2013 default $R_p = 50$ nm ($\Omega = 3.667$):

| Branch | Mask | Cell indices (npts=300) | θ range |
|---|---|---|---|
| A (Li-poor) | `c_p ≤ c_sp1` | 0–49 | 0.00–0.163 |
| B (mixed) | `c1_star ≤ c_p ≤ c2_star` | 9–291 | 0.03–0.97 |
| C (Li-rich) | `c_p ≥ c_sp2` | 251–299 | 0.837–1.00 |

Note: these indices differ from the original hardcoded Ω=3 values (62, 237, etc.) because the size-dependent Ω widens the miscibility gap.

**Key advantage of single-domain masking**: Because all three branches live on one shared FV mesh with a zero-flux (ghost = 0) boundary at the left edge of the domain, the transition fluxes are **directly extractable** via the telescoping integral. For example, $F_a(0) = 0$ exactly, so $J_{A \to B} = F_a(c_{sp1})$ — just the FV-computed flux at the interior cell edge corresponding to $c_{sp1}$. Similarly, the flux at $c_1^*$, $c_{sp2}$, and $c_2^*$ can be read off directly. In the three-subdomain architecture (`PyBaMM_acm`), this clean extraction is not possible for branch B: the FV discretization of each subdomain only provides zero-flux at the *physical* boundaries of that subdomain, and there is no mechanism to impose that the flux at $c_1^*$ and $c_2^*$ (the interior edges of B's subdomain) is zero — those edges carry the actual transition flux, but their value must be reconstructed rather than read off the shared stencil.

**Trade-off**: The single-domain approach is simpler to set up and gives natural access to all transition fluxes, but mask multiplication introduces subtle numerical artifacts at transition boundaries — see Challenge 4 below.

### 2.3 State Variables

Three `pybamm.Variable` objects defined on the shared mesh:

- **`g_a(c_p, x, t)`** — Branch A PDF (63 active cells)
- **`g_b(c_p, x, t)`** — Branch B PDF (259 active cells)
- **`g_c(c_p, x, t)`** — Branch C PDF (63 active cells)

Each carries domain hierarchy `{primary: "CCPM positive particle concentration", secondary: "positive electrode", tertiary: "current collector"}` for coupling into macro DFN.

Derived variables:

| Variable | Definition | Physical Meaning |
|---|---|---|
| $m_x$ | $\int g_x \, dc_p$ | Mass fraction in branch $x$ |
| $\theta_{\text{CCPM}}$ | $\int (g_a + g_b + g_c) \cdot (c_p / c_{\max}) \, dc_p$ | Average cathode stoichiometry |
| $J_{A \to B}$ | Flux of $g_a$ at $c_{sp1}$ (when $R > 0$) | Discharge transition rate |
| $J_{B \to C}$ | Flux of $g_b$ at $c_{2}^*$ (when $R > 0$) | Discharge transition rate |

### 2.4 PDE System and Advective Flux Construction

Each branch obeys a first-order **hyperbolic advection equation** with source terms:

$$
\frac{\partial g_x}{\partial t} = -\nabla \cdot F_x + S_x
$$

The spatial coordinate is lithium concentration $c_p$ (not position), and the "velocity" is the lithiation rate $R_x(c_p)$, which can be positive (lithiating — mass moves toward higher $c_p$) or negative (delithiating — mass moves toward lower $c_p$). Crucially, $R_x$ can change sign *within* a branch (see Challenge 4), so the flux must handle bidirectional advection at every cell.

#### Why Upwinding is Necessary

For a scalar conservation law $\partial_t g + \partial_c (g \cdot R) = 0$, central differencing of the advective flux $g \cdot R$ is unconditionally unstable (odd-even oscillations). The standard remedy is **upwind biasing**: at each cell edge, the PDF value $g$ is taken from the cell that is *upstream* with respect to the local flow direction. PyBaMM provides `pybamm.Upwind(g)` and `pybamm.Downwind(g)` operators that, on a 1D FV mesh of $n$ cells, produce $(n+1)$-edge vectors by taking the left-side and right-side cell values at each edge, respectively.

The advective flux at cell edges is then split by the sign of $R$:

$$
F_x = \text{Upwind}(g_x) \cdot R^+ + \text{Downwind}(g_x) \cdot R^-
$$

where $R^+ = \max(R, 0)$ and $R^- = \min(R, 0)$. When $R > 0$ at an edge, the flow is leftward-to-rightward and the Upwind (left-cell) value of $g$ is used; when $R < 0$, the Downwind (right-cell) value is used. This ensures that information always propagates in the correct characteristic direction.

#### Smooth Max/Min Approximation

PyBaMM's `Upwind`/`Downwind` operators produce symbolic expression trees that are evaluated by the CasADi solver. The hard `Maximum(R, 0)` and `Minimum(R, 0)` operators are non-differentiable at $R = 0$, which causes the implicit DAE solver (IDA/IDAS via CasADi) to struggle with Jacobian computation at zero-crossings of $R$.

The implementation replaces them with smooth approximations:

```python
R_pos = pybamm.smooth_max(R, 0, 100)   # ≈ max(R, 0)
R_neg = pybamm.smooth_min(R, 0, 100)   # ≈ min(R, 0)
```

The `smooth_max` and `smooth_min` functions in PyBaMM are defined as:

$$
\text{smooth\_max}(a, b, k) = \frac{1}{2}\left(\sqrt{(a - b)^2 + \sigma} + a + b\right), \quad \sigma = k^{-2}
$$
$$
\text{smooth\_min}(a, b, k) = \frac{1}{2}\left(a + b - \sqrt{(a - b)^2 + \sigma}\right), \quad \sigma = k^{-2}
$$

With $k = 100$, the smoothing parameter $\sigma = 10^{-4}$. This introduces a quantifiable approximation error:

| $R$ value | Exact $\max(R,0)$ | `smooth_max(R,0,100)` | Error |
|---|---|---|---|
| 0 | 0 | $\frac{1}{2}\sqrt{\sigma} = 0.005$ | $+0.005$ |
| $+1$ | 1 | $\frac{1}{2}(\sqrt{1 + 10^{-4}} + 1) \approx 1.000025$ | $+2.5 \times 10^{-5}$ |
| $-1$ | 0 | $\frac{1}{2}(\sqrt{1 + 10^{-4}} - 1) \approx 2.5 \times 10^{-5}$ | $+2.5 \times 10^{-5}$ |
| $\pm 0.01$ | $0.01$ or $0$ | adds $\sim 10^{-4}$ residual | $\sim 10^{-4}$ |

The worst-case error is at $R = 0$, where `smooth_max(0, 0, 100) = 0.005` instead of 0. This means that at the $R_a$ zero-crossing (Challenge 4, around cell 61), both the upwind and downwind terms contribute a small spurious flux of order $g \times 0.005$. For typical PDF values $g \sim 10^{-4}$, this gives a spurious flux of $\sim 5 \times 10^{-7}$ mol/(m³·s) — small compared to the physical flux ($\sim 10^{-3}$) but nonzero. Over $10^5$ seconds of C/30 simulation, the integrated spurious mass transport is $\sim 0.05$, which is $\sim 5\%$ of total mass — a non-negligible contribution to the traffic jam dynamics.

Increasing $k$ sharpens the approximation (error $\propto k^{-1}$ at $R = 0$) but increases the Jacobian condition number near the zero-crossing, making the solver work harder. The value $k = 100$ was chosen as a pragmatic balance between accuracy and solver robustness.

### 2.5 Branch Transitions via Divergence Telescoping (Clarke 2026 Eq. 18–20)

The central numerical question of the CCPM is: how do we compute the rate of mass leaving one branch and entering another? Clarke 2026 Eq. 18–20 define the transition magnitudes as boundary fluxes at the spinodal/transition concentrations. In a finite-volume framework, these boundary fluxes are not directly available as solver outputs — what the FV discretization produces is the *divergence* of the flux at cell centres. The single-domain approach solves this elegantly via the **telescoping property of the divergence integral**.

#### Derivation

Consider the integral of $\nabla \cdot F_a$ (the divergence of branch A's advective flux) over the concentration range $[0, c_{sp1}]$, which corresponds to branch A's active cells (0–62):

$$
J_{A \to B} = \int_0^{c_{sp1}} \nabla \cdot F_a \, dc_p
$$

By the fundamental theorem of calculus (the FV discrete analogue: summing cell divergences = net boundary flux), this telescopes to:

$$
J_{A \to B} = F_a(c_{sp1}) - F_a(0)
$$

Now, $F_a(0)$ is the advective flux at the left wall of the entire domain. Because we enforce **ghost = 0** at this boundary (see Section 2.6), the upwind stencil at edge 0 sees ghost = 0 from the left and $g_a[0]$ from the right. The flux becomes:

$$
F_a(0) = \underbrace{\text{Upwind}(g_a)}_{= 0 \text{ (ghost)}} \cdot R^+ + \underbrace{\text{Downwind}(g_a)}_{= g_a[0]} \cdot R^-
$$

But we also enforce the `zero_flux` BC type, which multiplies the boundary flux by 0 via a sparse diagonal mask matrix in the FV divergence operator (see Section 2.6). Therefore $F_a(0) = 0$ **exactly**, and:

$$
J_{A \to B} = F_a(c_{sp1})
$$

The transition magnitude is simply the FV-computed edge flux at $c_{sp1}$. In the implementation, `pybamm.Integral(pybamm.div(F_a) * mask_a, c_p)` computes this by summing the divergence over cells 0–62 and letting it telescope.

The same logic applies to all four transitions:

| Transition | Integral | Telescopes to | Direction |
|---|---|---|---|
| $J_{A \to B}$ | $\int_0^{c_{sp1}} \nabla \cdot F_a \, dc_p$ | $F_a(c_{sp1})$ | Discharge |
| $J_{B \to C}$ | $\int_0^{c_{2}^*} \nabla \cdot F_b \, dc_p$ | $F_b(c_2^*)$ (since $F_b(0) = 0$) | Discharge |
| $J_{B \to A}$ | $\int_{c_1^*}^{c_{\max}} \nabla \cdot F_b \, dc_p$ | $-F_b(c_1^*)$ (since $F_b(c_{\max}) = 0$) | Charge |
| $J_{C \to B}$ | $\int_{c_{sp2}}^{c_{\max}} \nabla \cdot F_c \, dc_p$ | $-F_c(c_{sp2})$ (since $F_c(c_{\max}) = 0$) | Charge |

#### Why This Only Works in the Single-Domain Architecture

The telescoping extraction relies on the fact that the divergence summation starts (or ends) at a **known-zero flux boundary**. In the single-domain approach, all branch PDFs share the same $[0, c_{\max}]$ domain, and the zero-flux condition at the domain walls ($c_p = 0$ and $c_p = c_{\max}$) is imposed once and applies to all. The masks then select the desired subrange.

In the three-subdomain architecture (`PyBaMM_acm`), branch B has its own domain $[c_1^*, c_2^*]$ with its own boundary conditions. To extract $J_{B \to C}$ we would need $F_b(c_2^*)$, which is the flux at B's *right physical boundary*. But transitional flux is precisely what flows through that boundary — we cannot simultaneously set it to zero (to make telescoping work) and use it to carry the transition. In that architecture, the transition flux must be reconstructed from edge-extrapolated $R$ and boundary-cell $g$ values, which is less direct.

#### Mass Deposition

Extracted mass is deposited into the receiving branch via a **normalized FV box function** — a single cell-width rectangle at the transition concentration:

$$
\delta_{\text{box}}(c_p) = \frac{\mathbb{1}(c_{\text{edge}} \leq c_p \leq c_{\text{edge}} + \Delta c)}{\int \mathbb{1} \, dc_p}
$$

The source term is $S = J \cdot \delta_{\text{box}}$. Since $\int \delta_{\text{box}} \, dc_p = 1$ by construction, $\int S \, dc_p = J$ exactly — mass is conserved.

For **discharge**, mass exits A at $c_{sp1}$ and is deposited into the first B cell above $c_{sp1}$ (cell 63). For **charge**, mass exits C at $c_{sp2}$ and is deposited into the last B cell below $c_{sp2}$ (cell 236). The box functions `cell_delta(c_edge_left)` and `cell_delta_charge(c_edge_right)` handle these two cases.

#### Mode Separation

`_get_transition_sources()` accepts `mode="discharge"` or `mode="charge"`:
- **Discharge**: only $J_{A \to B}$ and $J_{B \to C}$ source terms are constructed
- **Charge**: only $J_{C \to B}$ and $J_{B \to A}$ source terms are constructed

The inactive-direction transitions are set to `pybamm.Scalar(0)` — never built in the symbolic expression tree, not merely gated by a sigmoid. This eliminates the residual leakage from approximate Heaviside gating (see Challenge 7).

### 2.6 Boundary Conditions: Ghost = 0 via a Custom `zero_flux` BC Type (done with the help of Claude Opus 4.6 CoPilot)

PyBaMM's `Upwind`/`Downwind` operators require **Dirichlet boundary conditions** to construct ghost cells. The standard PyBaMM FV framework does not support zero-flux BCs for upwind operators — it only offers `"Dirichlet"` and `"Neumann"`, and Neumann BCs are not compatible with the ghost-cell stencil that Upwind/Downwind need.

However, the CCPM's physical walls (left of branch A, right of branch C) must be impermeable: no mass should leak out of the domain. A standard Dirichlet BC with $g_{\text{boundary}} = 0$ would set the ghost cell to $g_{\text{ghost}} = 2 \times 0 - g[\text{last}] = -g[\text{last}]$, which introduces a nonzero (and sign-inverted) ghost value. Combined with the upwind stencil, this generates a boundary flux that drains mass — the source of the 6% mass loss described in Challenge 1.

#### Solution: Adding a Custom `zero_flux` BC Type

A new BC type `"zero_flux"` was added to PyBaMM's `FiniteVolume.divergence()` method (commit `2db3461`). It operates in two stages:

1. **Ghost-cell construction** (for the Upwind/Downwind stencil): `zero_flux` is treated identically to `"Dirichlet"` with $g_{\text{boundary}} = 0$. This sets:
$$
g_{\text{ghost}} = 2 \times (0) - g[\text{boundary}] = -g[\text{boundary}]
$$
The ghost cell therefore mirrors the boundary cell with opposite sign. The upwind stencil at the boundary edge computes a flux as usual — this flux may be nonzero.

2. **Divergence mask** (kills the boundary flux): Before the divergence matrix $D$ is applied to the $(n+1)$-edge flux vector, a sparse diagonal mask matrix $M$ is applied that sets the boundary-edge entries to zero:
$$
M = \text{diag}(0, 1, 1, \ldots, 1, 0)
$$
The masked flux vector $\tilde{F} = M \cdot F$ has identically zero entries at edges 0 and $n$. The divergence then computes $D \cdot \tilde{F}$, and since cells 0 and $n-1$ only see their adjacent boundary edge (which is now zero), the boundary flux contribution to those cells is exactly zero.

The result: the interior upwind stencil is completely unchanged (no loss of advection accuracy), but the boundary cells see zero flux from the wall — a true impermeable boundary.

#### Boundary Condition Specification

In the code:
```python
self.boundary_conditions = {
    g_a: {"left": (zero, "zero_flux"), "right": (zero, "zero_flux")},
    g_b: {"left": (zero, "zero_flux"), "right": (zero, "zero_flux")},
    g_c: {"left": (zero, "zero_flux"), "right": (zero, "zero_flux")},
}
```

Note that all three PDFs share the same full domain $[\varepsilon_c, c_{\max} - \varepsilon_c]$, so the zero-flux condition applies at the physical walls ($c_p = \varepsilon_c$ and $c_p = c_{\max} - \varepsilon_c$). At *interior* transition edges ($c_{sp1}$, $c_1^*$, etc.), the flux is nonzero — this is by design, as these fluxes are precisely what the telescoping integral extracts (Section 2.5).

### 2.7 Interface Kinetics

**File**: `src/pybamm/models/submodels/interface/kinetics/ccpm_positive_interface.py` (180 lines)

The transfer current density for each branch follows Clarke 2026 Eq. 10–14:

$$
j_{\text{tr},k}(c_p) = j'_{p0} \cdot \left(\frac{c_e}{c_{e,\text{init}}}\right)^{0.5} \cdot i_{0,k}(c_p) \cdot \sinh\!\left(\frac{F\eta + \mu_k(c_p)}{2RT}\right)
$$

where:

| Branch | Exchange current $i_{0,k}$ | Chemical potential $\mu_k$ |
|---|---|---|
| A (single-phase) | $\theta_p^{0.5} (1 - \theta_p)^{0.5}$ — varies with $c_p$ | $RT[\ln(\theta_p / (1 - \theta_p)) + \Omega(1 - 2\theta_p)]$ |
| B (mixed-phase) | $\theta_{1}^{*\,0.5} (1 - \theta_{1}^*)^{0.5}$ — **constant** | $\mu_{\text{sep}} \approx 0$ |
| C (single-phase) | Same as A | Same as A |

Lithiation rates: $R_k = -\beta \cdot j_{\text{tr},k}$, where $\beta = 3 / (F \cdot 0.85 \cdot R_p)$.

The total interfacial current coupling back to the DFN macroscopic equations:

$$
j_{\text{tot}} = \int \left( \text{mask}_a \cdot g_a \cdot j_{\text{tr},a} + \text{mask}_b \cdot g_b \cdot j_{\text{tr},b} + \text{mask}_c \cdot g_c \cdot j_{\text{tr},c} \right) dc_p
$$

### 2.8 Open-Circuit Potential

**File**: `src/pybamm/models/submodels/interface/open_circuit_potential/ccpm_ocp.py` (45 lines)

$$
U_{\text{eq,eff}}(\theta) = U_{eq,p0} - \frac{RT}{F}\left[\ln\!\left(\frac{\theta}{1 - \theta}\right) + \Omega(R_p)(1 - 2\theta)\right]
$$

with $U_{eq,p0} = 3.397\,\text{V}$ (Prada 2013). The regular solution parameter is now **size-dependent** (see Section 2.11):

```python
omega = 4.5 - (2.5 / 0.18e9) * 3 / self.param.p.prim.R
```

This is the symbolic (PyBaMM expression tree) version — it references the particle radius parameter so that the OCP formula automatically adapts when `parameter_values["Positive particle radius [m]"]` changes.

The entropic term $dU/dT = 0$ (not yet implemented).

### 2.9 Initial Conditions

Two initialization paths:

| Start Branch | Distribution | Formula |
|---|---|---|
| A (delithiated) | Gamma(2, λ) | $g_a(c) \propto (c - c_{\min}) \cdot (c_{sp1} - c) \cdot \exp(-(c - c_{\min})/\lambda) \cdot \mathbb{1}(c \leq c_{sp1})$ |
| C (lithiated) | Tight Gaussian | $g_c(c) \propto \exp\!\left(-\frac{(c - c_{\text{init}})^2}{2(2 \Delta c)^2}\right)$ |

#### Why Gamma(2, λ) for Branch A Instead of Gaussian

With Prada 2013 parameters: $c_{\text{init}} = 86.66\,\text{mol/m}^3$, $c_{\max} = 22806\,\text{mol/m}^3$, $\theta_0 = 0.0038$, $c_{\min} = \varepsilon_c \approx 2.28 \times 10^{-4}\,\text{mol/m}^3$.

The initial stoichiometry is extremely low ($\theta_0 = 0.0038$), meaning $c_{\text{init}}$ is very close to the left domain boundary $c_{\min}$: only 86.66 mol/m³ from a wall at $\approx 0$. A Gaussian centered at $c_{\text{init}}$ would extend symmetrically to both sides, but the domain is bounded below at $c_{\min}$. This creates a fundamental problem:

**Gaussian width vs. mesh resolution constraint**: The uniform mesh has $\Delta c \approx 76$ mol/m³. For a Gaussian to be adequately resolved on this mesh, it needs $\sigma \geq 2\Delta c \approx 152$ mol/m³ (at least 4–5 cells across the main bell). But $c_{\text{init}} = 86.66$ mol/m³, so the Gaussian center is only $86.66 / 152 \approx 0.57\sigma$ from the domain wall. The left $3\sigma$ point would be at $86.66 - 456 = -369$ mol/m³ — far below $c_{\min}$.

**Mass below the wall**: A Gaussian $\mathcal{N}(86.66, 152^2)$ truncated at $c_{\min} = 0$ loses:

$$
\Phi\!\left(\frac{c_{\min} - c_{\text{init}}}{\sigma}\right) = \Phi(-0.57) \approx 28.4\%
$$

of its total mass to the left of the wall. Truncation removes this mass and shifts the effective mean rightward to $\approx 205$ mol/m³ (more than double $c_{\text{init}}$). Renormalizing the truncated Gaussian gives $\theta_{\text{eff}} \approx 0.009$ instead of the target $\theta_0 = 0.0038$ — a **137% error** in initial stoichiometry.

**Reducing σ doesn't help**: To keep 99% of the mass above $c_{\min}$, we need $c_{\text{init}} \geq 2.33\sigma$, giving $\sigma \leq 37.2$ mol/m³ — less than half a cell width. Such a Gaussian is sub-grid: it concentrates all mass into $\sim$1–2 cells, defeating the purpose of a smooth initial distribution and creating steep gradients that stress the solver.

**The Gamma(2, λ) solution**: The Gamma distribution shifted to $c_{\min}$ is:

$$
g(c) \propto (c - c_{\min})^{k-1} \cdot e^{-(c - c_{\min})/\lambda}, \quad k = 2
$$

This is **identically zero** at $c = c_{\min}$ (via the $(c - c_{\min})$ prefactor) and naturally right-skewed. No truncation is needed. The mean is $c_{\min} + k\lambda = c_{\min} + 2\lambda$, giving $\lambda = (c_{\text{init}} - c_{\min})/2 \approx 43.3$ mol/m³. The mode (peak) is at $c_{\min} + \lambda \approx 43.3$ mol/m³, and the distribution decays exponentially beyond the mean. On the 300-point mesh, this shape spans $\sim$3–4 cells with smooth leading and trailing edges — well resolved and solver-friendly.

The additional $(c_{sp1} - c)$ damping factor ensures compact support: the PDF tapers to zero at the spinodal boundary $c_{sp1}$ rather than leaving an exponential tail extending into branch B's domain.

#### Two-Pass λ Correction

The analytical $\lambda_0 = (c_{\text{init}} - c_{\min})/2$ is exact for the continuous Gamma(2, λ). On the discrete 300-point mesh, numerical integration introduces a small bias in the mean ($\sim$0.3% error on $\theta_0$). A two-pass correction eliminates this:

1. Build trial PDF with $\lambda_0$; compute discrete mean $\mu_{\text{trial}} = \int g \cdot c \, dc_p / \int g \, dc_p$
2. Rescale: $\lambda = \lambda_0 \cdot (c_{\text{init}} - c_{\min}) / (\mu_{\text{trial}} - c_{\min})$
3. Rebuild PDF with corrected $\lambda$

This ensures $\int g \cdot c \, dc_p / \int g \, dc_p = c_{\text{init}}$ to machine precision after normalization.

#### Branch C Initialization

For a lithiated start ($\theta_0 = 0.90$), $c_{\text{init}} = 0.90 \times 22806 = 20525$ mol/m³ is deep inside branch C's domain $[c_{sp2}, c_{\max}]$ — far from both boundaries. A Gaussian works perfectly here:

$$
g_c(c) \propto \exp\!\left(-\frac{(c - c_{\text{init}})^2}{2\sigma^2}\right), \quad \sigma = 2\Delta c \approx 152\,\text{mol/m}^3
$$

The distribution is zero within $\sim$4–5 cells of the center, well away from both $c_{sp2}$ and $c_{\max}$. No truncation or special treatment is needed.

### 2.10 Key Numerical Constants

| Constant | Value | Source |
|---|---|---|
| $c_{\max}$ | 22 806 mol/m³ | Prada 2013 |
| $\varepsilon_c$ | $10^{-8} \times c_{\max}$ | Domain truncation |
| $n_{\text{pts}}$ | 300 (default, configurable) | DFN_CCPM constructor `npts` kwarg |
| $\Delta c$ | $\approx c_{\max} / n_{\text{pts}}$ | Uniform grid spacing |
| $U_{eq,p0}$ | 3.397 V | Prada 2013 |
| $j'_{p0}$ | 0.948 A/m² | Prada 2013 |
| $\Omega$ | Size-dependent (see Section 2.11) | Ferguson & Bazant 2014 |
| $\beta$ | $3/(F \cdot 0.85 \cdot R_p)$ | Current-to-rate; 0.85 is a **debug** factor (nominal = 1.0) |

**Branch boundaries** are now computed dynamically from Ω via `compute_branch_constants()` and snapped to the nearest FV cell edge. Example values for Prada2013 defaults ($R_p = 50$ nm, $\Omega = 3.667$, npts=300):

| Constant | θ (analytical) | Cell edge (npts=300) | θ (snapped) |
|---|---|---|---|
| $c_{sp1}$ | 0.1632 | 49 | 0.1633 |
| $c_{sp2}$ | 0.8368 | 251 | 0.8367 |
| $c_1^*$ | 0.0301 | 9 | 0.0300 |
| $c_2^*$ | 0.9699 | 291 | 0.9700 |

### 2.11 Size-Dependent Regular Solution Parameter Ω(R_p)

The regular solution parameter $\Omega$ controlling the LFP miscibility gap is no longer hardcoded to 3. Following Ferguson & Bazant (2014, *Electrochim. Acta* 146, 89–97), $\Omega$ depends on the particle radius:

$$
\Omega(R_p) = 4.5 - \frac{2.5}{0.18 \times 10^9} \cdot \frac{3}{R_p}
$$

where $R_p$ is in metres. This captures the experimental observation that smaller LFP particles have a narrower miscibility gap (lower Ω → spinodal and binodal concentrations move closer together), eventually suppressing phase separation entirely for particles below ~36 nm ($\Omega < 2$).

**Example values:**

| $R_p$ [nm] | $\Omega$ | $\theta_{sp1}$ | $\theta_{sp2}$ | $\theta_{b1}$ | $\theta_{b2}$ |
|---|---|---|---|---|---|
| 50 (Prada2013) | 3.667 | 0.163 | 0.837 | 0.030 | 0.970 |
| 100 | 4.083 | 0.139 | 0.861 | 0.018 | 0.982 |
| 200 | 4.292 | 0.129 | 0.871 | 0.013 | 0.987 |
| 36.1 | 2.003 | 0.499 | 0.501 | — | — |

**Implementation**: The computation is performed in three places:

1. **`CCPMPositiveParticle.compute_branch_constants(R_p_float, npts, c_max_float)`** — static method that takes Python floats, computes Ω numerically via the formula above, derives spinodals ($\theta_{sp} = (1 \pm \sqrt{1-2/\Omega})/2$) and binodals (via `scipy.optimize.brentq` on $\ln(\theta/(1-\theta)) + \Omega(1-2\theta) = 0$), snaps all four boundaries to the nearest FV cell edge, and returns them as `pybamm.Scalar` objects.

2. **`CCPMPositiveInterface.__init__`** — calls `compute_branch_constants` with the same floats to obtain Ω and boundary constants for the kinetics equations.

3. **`ccpm_ocp.py`** — uses the symbolic expression `omega = 4.5 - (2.5/0.18e9)*3/self.param.p.prim.R` so that Ω participates in the PyBaMM expression tree and is automatically differentiated for the Jacobian.

**DFN_CCPM constructor flow**: `parameter_values` is required at construction time. The constructor extracts `R_p_float = float(parameter_values["Positive particle radius [m]"])` and `c_max_float = float(parameter_values["Maximum concentration in positive electrode [mol.m-3]"])`, then passes these to submodel constructors. This works because Prada2013 defines $R_p$ as a constant (not a function of state), so `float()` extraction is valid.

---

## 3. Development Timeline

| Date | Commit | Milestone |
|---|---|---|
| Mar 14 | `db70146` | Skeleton `DFN_CCPM` — pure DFN inheritance, no new physics |
| Mar 16 | `6864ea6`–`cd48f7c` | Added dummy $g_a$, $g_b$, $g_c$ variables; first OCP from stoichiometry; still inheriting `FickianDiffusion` |
| Mar 17 | `8acb18a`–`812f99f` | First running simulation with trivial branch init. Mass drift observed immediately |
| Mar 23 | `3f9db13` | **"Works correctly for C/30 in 50 seconds"** — first successful short discharge |
| Mar 26 | `0b8b5d1` | Removed `FickianDiffusion` inheritance; standalone particle submodel |
| Mar 26 | `96cf97a`–`bf6118e` | Branch transition sources added; simulation time became "too long"; mask-truncated flux attempted ("NOT SURE IF THIS IS CORRECT") |
| Mar 27 | `4590eff`–`3f152f9` | Mass loss battle: started at 6% loss in 5 s at C/2; fixed by forcing R = 0 in ghost cell → conserves mass for 5 s |
| Mar 28 | `2db3461` | Added `zero_flux` BC type to PyBaMM's FV discretization module |
| Mar 30 | `d81f1f8`–`d3932a9` | Mass-conservative transitions via integrated-divergence sources; replaced Gaussian deposit with single-cell FV box |
| Mar 31 | `3bcd4a9`–`e098ba9` | Zero-flux BC for branch B; telescoping integral correction; pickle fix; bidirectional transitions; branch C init; domain snapping |
| Apr 1 | `d655b32` | Updated parameters to Prada 2013: $U_{eq,p0} = 3.397$ V, $j'_{p0} = 0.948$ A/m² |
| Apr 2 | `d27b5bc` | Removed mask from $R$; fixed sinh catastrophic cancellation |
| Apr 2 | `4f8ca4b` | Charge-mode delta functions; OCP diagnostics; mode-split test scripts (HEAD) |

---

## 4. Challenges Encountered and How They Were Resolved

### Challenge 1: Mass Conservation Failure (6% loss in 5 seconds)

**When**: Mar 27, commits `4590eff`–`1954a99`

**Problem**: The first dynamic simulation at C/2 rate lost **6% of total PDF mass** ($\sum m_x$) within just 5 seconds of simulation time. Mass should be conserved exactly ($dm/dt = 0$ when summing all three branches).

**Root cause**: PyBaMM's finite-volume discretization creates ghost cells at domain boundaries. The upwind stencil at the left boundary of branch A (and right boundary of C) saw nonzero ghost values, creating a spurious outward flux — mass literally leaked out of the computational domain through the ghost cells.

**Attempted fix 1** (commit `1954a99`): Flux BC compensation — manually subtract the leaking boundary flux. Reduced loss but did not eliminate it: still **6% loss at 5 s**.

**Successful fix** (commit `3f152f9`): Force $R = 0$ in the ghost cell. Ghost node with $R = 0$ produces zero flux through the boundary, regardless of the PDF value. Mass conservation achieved for 5 s.

**Later refinement** (commit `2db3461`): Added a proper `zero_flux` boundary condition type directly into PyBaMM's FV discretization so that the boundary flux is analytically zero, rather than relying on $R = 0$ hacks.

**Final result**: With ghost = 0 BCs and single-cell box deposits, mass drift at **1C rate** is $+1.08 \times 10^{-6}$ (**0.0001%**). At C/30 the prior Dirichlet $g = 0$ approach had given 292–607% drift; the current ghost = 0 approach eliminated this entirely.

---

### Challenge 2: Gaussian Source Deposition Smearing

**When**: Mar 30, commit `d3932a9`

**Problem**: Branch transition sources were originally deposited using a regularized Gaussian kernel $\delta(c - c_{\text{target}}) \approx \exp(-\Delta c^2 / 2\sigma^2) / (\sigma\sqrt{2\pi})$ with $\sigma = 1.5 \cdot \Delta c_{\text{local}}$. This smears the deposited mass across ~5 cells.

**Numerical impact**: The Gaussian tails extend beyond the branch boundary (e.g., mass deposited at the B boundary spills back into A's cells), breaking branch isolation and making mass accounting ambiguous. With 300 uniform cells ($\Delta c \approx 76$ mol/m³), $\sigma \approx 114$ mol/m³ — the Gaussian half-width covers 1.5 cells in each direction.

**Fix**: Replaced Gaussian kernel with a **single-cell FV box function**: all deposited mass goes into exactly one cell adjacent to the transition boundary. The box is normalized so $\int \delta_{\text{box}} \, dc_p = 1 / \Delta c$, and the source is $S = J \cdot \delta_{\text{box}}$. Mass conservation is exact by construction: $\int S \, dc_p = J$.

---

### Challenge 3: Catastrophic Cancellation in sinh Argument

**When**: Apr 2, commit `d27b5bc`

**Problem**: The transfer current density involves $\sinh\!\left(\frac{F\eta + \mu_k}{2RT}\right)$ where $\eta = \phi_s - \phi_e - U_{eq,p0}$. In the naïve implementation, $\phi_s$ and $\phi_e$ were first broadcasted individually from the macroscopic $x$-grid (20 nodes) to the 300-cell concentration grid via `PrimaryBroadcast`, and $U_{eq,p0}$ was subtracted afterwards on the concentration grid. This triggers catastrophic cancellation.

#### Worked Example (IEEE 754 double precision, 16 significant digits)

At a representative positive electrode node during C/30 discharge:
- $\phi_s = 3.40200000000000\mathbf{0000}$ V (solid-phase potential, 16 digits)
- $\phi_e = -0.00300000000000\mathbf{0000}$ V (electrolyte potential)
- $U_{eq,p0} = 3.39700000000000\mathbf{0000}$ V (reference OCP)
- $T = 298.15$ K, so $2RT/F = 0.05139$ V

The physical overpotential is: $\eta = \phi_s - \phi_e - U_{eq,p0} = 3.402 - (-0.003) - 3.397 = 0.008$ V

**Naïve approach — broadcast first, subtract later:**

After broadcasting $\phi_s$ and $\phi_e$ to the concentration grid, each cell computes:

$$
\frac{F(\phi_s - \phi_e)}{2RT} = \frac{3.405}{0.05139} = 66.2502\ldots
$$
$$
\frac{F \cdot U_{eq,p0}}{2RT} = \frac{3.397}{0.05139} = 66.0944\ldots
$$
$$
\frac{F\eta}{2RT} = 66.2502\ldots - 66.0944\ldots = 0.1557\ldots
$$

The subtraction $66.25 - 66.09$ cancels the 2 leading digits. In IEEE 754 double precision, each of the two operands has ~16 significant digits, but after subtracting two numbers that agree in their first 2 digits, the result retains only ~14 significant digits. Those "lost precision" values are further combined with $\mu_k/(2RT)$ and passed through $\sinh$, which amplifies small errors exponentially. Hence the accumulated floating point noise might lead to errors in j_tr computation.

More specifically, during the flat LFP plateau, $\phi_s - \phi_e$ can be as close as 3.3975 V to $U_{eq,p0} = 3.397$ V, making $\eta \approx 0.0005$ V and the ratio:

$$
\frac{F\eta}{2RT} = \frac{0.0005}{0.05139} \approx 0.01
$$

computed as $66.097\ldots - 66.094\ldots = 0.003\ldots$ — now the first **4 digits** cancel, leaving only ~12 reliable digits before any further arithmetic.

**Fix — subtract first, broadcast the small result:**

```python
eta = phi_p - phi_e - U_eq_p0          # macroscopic, O(0.01 V) — no cancellation here
eta_broadcast = PrimaryBroadcast(eta, "CCPM positive particle concentration")

# sinh argument on the concentration grid:
overpotential_term = (F * eta_broadcast + mu) / (2 * R * T_broadcast)
```

The subtraction $\phi_s - \phi_e - U_{eq,p0}$ is performed at the macroscopic scale on the $x$-grid, where it involves the same three scalar values and produces $\eta = 0.008$ V directly — no large intermediate values, no cancellation. Only the small $\eta$ is then broadcast to the 300-cell grid. On the concentration grid, the sinh argument is $F\eta/(2RT) + \mu_k/(2RT) \approx 0.16 + \mu_k/(2RT)$, all well-conditioned. This is mathematically identical but avoids the subtraction of two $O(66)$ quantities.

---

### Challenge 4: Mass Stall at transition A->B

**When**: Apr 2 (diagnosed), still present at HEAD

**Problem**: During C/30 discharge, the final stochiometry was 0.50, with 60% of mass still in Branch A, concentrated close to c_sp1. It was observed that the lithiation rate $R_a(c_p)$ approaches zero and changes sign within branch A near $\theta \approx 0.204$ (cell 61 of 300). The primary cause of the stall is not the sign change itself, but the fact that $R_a$ becomes vanishingly small in magnitude across the entire upper portion of branch A as the PDF's mass front approaches the spinodal point $c_{sp1}$.

The physics: as particles lithiate and their concentration nears the spinodal, the chemical potential $\mu_a(c_p) = RT[\ln(\theta/(1-\theta)) + \Omega(1-2\theta)]$ grows steeply, increasingly opposing the macroscopic driving force $F\eta$. The resulting lithiation rate $R_a \propto \sinh((F\eta + \mu_a)/(2RT))$ becomes extremely small when $F\eta \approx -\mu_a$ — the electrochemical driving force and the thermodynamic resistance nearly cancel. As $R_a$ decays from its bulk value ($\sim 10^{-2}$ mol/(m³·s) at low $c_p$) to $\sim 10^{-6}$–$10^{-8}$ near the spinodal, a drop of 4–6 orders of magnitude, the advancing PDF front reaches this near-zero-$R$ region and mass cannot be pushed through to $c_{sp1}$ at any meaningful rate.

The sign change only matters to the extent that the tiny negative $R$ above the zero-crossing contributes a small reverse flux. But this reverse flux is negligible compared to the stall caused by the near-zero forward rate. If there were no mass in the cells above the zero-crossing (i.e., the PDF hadn't arrived there yet), the negative $R$ values would have no effect — they multiply $g_a \approx 0$, producing zero flux.

**Concrete numbers** (from `diagnose_stall.py`, C/30 discharge to 2 V):
- $R_a$ at the advancing PDF front (cells 55–62): drops from $\sim 10^{-3}$ to $\sim 10^{-7}$ mol/(m³·s) — a near-stagnation zone spanning ~8 cells (~600 mol/m³ in concentration space)
- $J_{A \to B}$ plateaus at $\sim 8.4 \times 10^{-6}$ (tiny constant trickle, set by the vanishing $R$ at the boundary)
- $dm_a / dt \approx -J_{A \to B}$ (only way mass leaves A is this slow boundary leak)
- At end of discharge: **60% of total mass stuck in branch A**, only **39% in C**
- $\theta_{\text{CCPM}}$ stalls at $\sim 0.51$ (should reach $\sim 0.7$ for full discharge)
- Voltage drops to 2 V purely through overpotential, not OCP — the cell "hits the wall" without completing the phase transition

**Why the stall persists**: On the flat LFP OCP plateau, the macroscopic overpotential $\eta$ barely changes as $\theta$ increases. Since $\eta$ remains approximately constant, the near-cancellation between $F\eta$ and $\mu_a(c_p)$ near the spinodal is **pinned** — the solver cannot self-correct by increasing $\eta$ because the flat OCP means $\eta$ doesn't need to change to maintain the applied current (the current is still supplied, just by the remaining mass at lower $c_p$ where $R_a$ is still appreciable). This is physically realistic: in real LFP electrodes, particles do get stuck in metastable single-phase states near the spinodal. However, it creates a numerical bottleneck because the model needs a mechanism to push mass through a region where the advection velocity is essentially zero.

**Mitigation (debug investigation)**: To confirm that the stall was caused by insufficient advection velocity at the transition edge rather than a fundamental model error, the effective particle radius was temporarily reduced via the debug scaling factor in $\beta = 3 / (F \cdot \text{factor} \cdot R_p)$. The Clarke 2026 formula uses $\beta = A / (F \cdot V) = 3 / (F \cdot R_p)$ for spherical particles (surface-to-volume ratio $A/V = 3/R_p$). The factor was decreased from 1.0 (nominal Prada 2013 radius $R_p = 50$ nm) to 0.85 (effective 42.5 nm).

A smaller effective radius yields a larger $\beta$, giving a faster lithiation rate $R_k = -\beta \cdot j_{\text{tr},k}$ for a given transfer current density. This increases the advection velocity across the concentration grid near $c_{sp1}$, partially compensating for the near-zero $R_a$ at the spinodal by scaling up its absolute value.


**Note on mask placement**: In an earlier version (commit `6de8c7d8d`, "mask R_a"), the mask was applied directly to R: `R_a_masked = R_a * mask_a`. This zeroed $R$ at cell 63, and the FV node-to-edge averaging at edge 63 then computed $(R[62] + 0)/2 = R[62]/2$ — halving the physical rate at the critical transition edge. This was fixed in commit `d27b5bc` by removing the mask from $R$. In the current implementation, the mask is applied **only to the divergence** (`rhs_a = -div(F_a) * mask_a`), not to $R$ itself. The flux $F_a = \text{adv\_flux}(g_a, R_a)$ uses the raw unmasked $R_a$, so node-to-edge interpolation at edge 63 sees the full physical $R$ from both neighboring cells. The mask on the RHS merely zeroes the divergence contribution outside branch A's active cells (63–299), preventing the ODE from updating $g_a$ where it should be zero. This resolves the halving artifact.

For now, we conclude that the model is fundamentally working and we move forward to the half cell validation model.
---

### Challenge 5: Mesh-Aligned Constants

**When**: Mar 31, commit `e098ba9`

**Problem**: The Clarke 2026 analytical values for spinodal and transition concentrations ($c_{sp1}/c_{\max} = 0.2113$, $c_1^*/c_{\max} = 0.0710$, etc.) do not fall exactly on cell edges of a 300-point uniform mesh. When a source deposits mass "at $c_{sp1}$", the FV box straddles two cells, splitting the deposited mass unpredictably.

**Example**: With 300 uniform cells, the raw $c_{sp1} = 0.2113 \times 22806 = 4818.8$ mol/m³. The nearest cell edges are at $\Delta c \times 63 = 4789.3$ and $\Delta c \times 64 = 4865.3$. The deposit target falls **between** edges 63 and 64, meaning the single-cell box straddles two cells.

**Fix**: Snap each constant to the nearest cell edge index. The snapped values are computed as $c = \varepsilon_c + (k/300) \times L$ where $L = c_{\max} - 2\varepsilon_c$:

| Constant | Clarke 2026 θ | Cell index $k$ | Exact $c$ [mol/m³] | Exact θ (= $c/c_{\max}$) | Δθ from Clarke |
|---|---|---|---|---|---|
| $c_1^*$ | 0.0710 | 21 | 1596.420196 | 0.070000009 | −0.14% |
| $c_{sp1}$ | 0.2113 | 63 | 4789.260132 | 0.210000006 | −0.06% |
| $c_{sp2}$ | 0.7887 | 237 | 18016.739868 | 0.789999994 | +0.02% |
| $c_2^*$ | 0.9290 | 279 | 21209.579804 | 0.929999991 | +0.01% |

The θ values are **not** exactly round numbers like 0.07 or 0.21, because $\varepsilon_c = 2.2806 \times 10^{-4}$ mol/m³ offsets the mesh slightly. The $O(10^{-8})$ deviations from the round fractions (e.g., $\theta_{sp1} = 0.210000006$ rather than $0.210000000$) are below the precision of any physical measurement but are recorded here for reproducibility. The snapping ensures that the cell edge used in the FV box deposit and divergence telescoping falls exactly on a mesh face, not between two faces.

All θ changes from Clarke 2026 are **< 0.15%**. In the sibling worktree `PyBaMM_acm`, these are computed via fixed-point iteration on the per-subdomain meshes; here, they are computed from integer cell indices directly.

---

### Challenge 6: Pickle Deserialization Across Python Sessions (done with the help of Claude Opus 4.6 CoPilot)

**When**: Mar 31, commit `3bcd4a9`

**Problem**: Saving a solved `Simulation` to pickle and reloading in a new Python process failed with `KeyError` or `ValueError: broadcast` errors. Two independent causes:

1. **Variable ID mismatch**: Python's `hash()` is seeded with `PYTHONHASHSEED` (randomized per process). When the pickle is loaded, `SymbolProcessor` calls `create_copy()` which generates new Variable objects with fresh hash-based IDs that don't match the frozen `y_slices` keys from the original process.

2. **Cached shape mismatch**: `domain_size()` uses `hash(domain_string) % 100` for non-standard domains like `"CCPM positive particle concentration"`. The cached `_saved_evaluate_for_shape` in old pickle nodes has shapes based on the old hash seed; new nodes get different sizes → broadcast error.

**Fix**: Added `__setstate__` to `Symbol` base class in `src/pybamm/expression_tree/symbol.py`:

```python
def __setstate__(self, state):
    self.__dict__.update(state)
    self.set_id()  # Recompute ID with current PYTHONHASHSEED
    self.__dict__.pop("_saved_evaluate_for_shape", None)  # Clear stale cache
```

**Verification**: All 18 CCPM variables extract correctly from old pickle; fresh round-trip gives zero diff.

---

### Challenge 7: Bidirectional Transition Flux Noise

**When**: Apr 2, commit `4f8ca4b`

**Problem**: When both charge and discharge transition fluxes are active simultaneously (all four $J_{A \to B}$, $J_{B \to A}$, $J_{B \to C}$, $J_{C \to B}$), the sigmoid gating `pybamm.sigmoid(0, J, k)` does not zero the inactive direction exactly.

PyBaMM's sigmoid is defined as:

$$
\text{sigmoid}(0, J, k) = \frac{1 + \tanh(k \cdot J)}{2}
$$

With $k = 10^4$, for a wrong-direction flux $J \sim -10^{-15}$ (numerical noise):

$$
\tanh(10^4 \times (-10^{-15})) \approx -10^{-11} \implies \text{sigmoid} \approx 0.5 - 5 \times 10^{-12}
$$

The gated flux is $J \times \text{sigmoid} \approx -10^{-15} \times 0.5 = -5 \times 10^{-16}$. Per time step this leakage is negligible, but it accumulates.

**Quantified leakage for C/30 discharge**: A C/30 discharge of an LFP cell with $c_{\max} = 22806$ mol/m³ runs for $\sim 1.08 \times 10^5$ seconds ($= 30 \times 3600$ s). The CasADi solver with `mode="safe"` uses adaptive time stepping with typically $\sim 10^3$–$10^4$ internal steps. With 4 sigmoid-gated fluxes each leaking $\sim 5 \times 10^{-16}$ per step:

$$
\text{Total leakage} \approx 4 \times 5 \times 10^{-16} \times 10^4 = 2 \times 10^{-11}
$$

relative to a total mass $m_{\text{tot}} = 1$. This is $\sim 2 \times 10^{-9}\%$ — negligible in isolation. However, the sigmoid also interacts with the `smooth_max`/`smooth_min` approximation (Section 2.4): at the $R_a$ zero-crossing, the smooth operators contribute a residual $\sim 0.005$ to both $R^+$ and $R^-$, generating a small bidirectional flux even where $R$ should be exactly zero. When these two sources of error compound over long simulations, the observed mass drift can reach $\sim 10^{-6}$–$10^{-5}$ (0.0001–0.001%), which was visible in the diagnostic output.

**Fix**: Introduced explicit `mode="discharge"` / `mode="charge"` parameter in `DFN_CCPM`. In discharge mode, only $J_{A \to B}$ and $J_{B \to C}$ source terms are constructed in the symbolic expression tree; $J_{B \to A}$ and $J_{C \to B}$ are set to `pybamm.Scalar(0)` — they are never built, not merely gated to approximately zero. This:
- Eliminates sigmoid leakage entirely for single-direction experiments
- Reduces the solver's symbolic expression count (from 4 fluxes to 2), modestly improving solve speed

Currently only `"discharge"` and `"charge"` modes are implemented; any other value raises a `ValueError`. A future bidirectional mode for cycling experiments (where all four fluxes would need to be active simultaneously) would require either reintroducing sigmoid gating with a tighter steepness or switching modes between experiment steps.

---

### Challenge 8: Fickian Diffusion Inheritance (Removed)

**When**: Mar 16–26, commits `6d6e956`–`0b8b5d1`

**Problem**: The initial implementation inherited from `FickianDiffusion` for code reuse. However, the CCPM physics is fundamentally different: it uses advection on a *concentration* grid rather than diffusion on a *radial* grid. The inheritance dragged in:
- Radial coordinate `r_p` geometry that conflicted with the `c_p` coordinate
- Diffusivity parameter lookups that don't exist for CCPM
- Boundary condition logic designed for spherical particle surfaces

**Fix** (commit `0b8b5d1`): Complete rewrite removing `FickianDiffusion` inheritance. `CCPMPositiveParticle` now inherits directly from `BaseParticle` and implements its own `get_fundamental_variables()`, `get_coupled_variables()`, `set_rhs()`, `set_boundary_conditions()`, and `set_initial_conditions()`. The particle submodel grew from ~50 lines (inherited) to 435 lines (standalone).

---

## 5. Test and Diagnostic Infrastructure

### 5.1 Test Scripts

| Script | Purpose | Rate | Duration | Key Metric |
|---|---|---|---|---|
| `quick_c10_100s.py` | Sanity check | C/10 | 100 s | PDF non-negativity, mass drift |
| `test_dynamics_plots_discharge.py` | Full discharge tracking | C/30 | → 3.18 V | All 4 transition fluxes |
| `test_dynamics_plots_charge.py` | Full charge tracking | C/30 | → 3.6 V | Reverse fluxes $J_{CB}$, $J_{BA}$ |
| `test_charge_c10_100s.py` | Charge validation | C/30 | → 3.6 V | Bidirectional transitions |
| `test_branch_c_init.py` | Branch C initial state | C/30 | 1 s | PDF shape, normalization, support |
| `test_half_cell_discharge.py` | Li metal ‖ LFP discharge | C/5 | 30000 s | Isolated cathode, Fig 7 OCP-shift plot |
| `test_half_cell_charge.py` | Li metal ‖ LFP charge | C/5 | 30000 s | Isolated cathode, Fig 7 OCP-shift plot |
| `test_dynamics.py` | Short dynamics | — | — | Mass conservation |
| `test_ccpm_refactor.py` | Integration | C/30 | 2100 s | End-to-end solver |

### 5.2 Diagnostic Scripts

| Script | Purpose | Key Findings |
|---|---|---|
| `diagnose_stall.py` | C/30 mass stagnation | 60% mass stuck in A; $\theta$ frozen at 0.51 |
| `diagnose_jtr_x.py` | $j_{\text{tr},a}$ spatial sign variation | Sign change at cells 58–62 across $x$-nodes |
| `postprocess3.py` | Raw state extraction | PDF arrays at all time steps |
| `postprocess_c30_210V.py` | OCP subplot analysis | Plotly-based interactive visualization |
| `postprocess_half_cell_fig7.py` | Clarke Fig. 7 replication | Callable `plot_fig7()` function; OCP-shift vs θ, 3-panel plot |
| `postprocess_half_cell_fig7_compare_potentials.py` | Potential comparison | Compare OCP across charge/discharge |
| `postprocess_half_cell_fig7_potentials.py` | Potential analysis | Detailed potential diagnostics |
| `postprocess_half_cell.py` | Half-cell post-processing | General half-cell solution analysis |
| `postprocess_half_cell_all_rates.py` | Multi-rate comparison | Compare half-cell results across C-rates |
| `postprocess_half_cell_voltage_all_rates.py` | Voltage comparison | Compare voltage curves across C-rates |

### 5.3 Pickle Workflow

Simulations are saved as `.pkl` files for post-processing without re-solving:
- `C30_2V_modeDch.pkl` — C/30 discharge to 2 V
- `C30_318V_modeDch_smallerRadius.pkl` — C/30 discharge to 3.18 V
- `C30_300s_modeChg.pkl` — C/30 charge for 300 s
- `half_cell_C30_discharge.pkl` — Half-cell discharge

---

## 6. Solver Performance

| Configuration | Rate | Duration | Wall Time | Drift | Notes |
|---|---|---|---|---|---|
| 300 pts, Casadi "safe" | 1C | full | — | 0.0001% | Excellent conservation |
| 300 pts, Casadi "safe" | C/30 | 50 s | — | — | First successful run |
| 50 pts exponential (old) | C/30 | 1000 s | 10.5 s | 0.5% | Pre-uniform mesh |
| 50 pts exponential (old) | C/30 | 10000 s | 30.5 s | 3.36% | $m_B \approx 0$ |
| 50 pts exponential (old) | C/30 | ~13775 s | — | — | Solver crash: corrector convergence |

The current 300-point uniform mesh with ghost = 0 BCs and FV box deposits has dramatically improved mass conservation compared to the early 50-point exponential mesh.

---

## 7. Open Items

1. **R_a traffic jam** (Challenge 4): The sign change in $R_a$ during discharge prevents full A → B transition at low C-rates. This is the most significant remaining obstacle.

2. **Parameter validation**: $j'_{p0} = 0.948$ A/m² is marked with a TODO comment. The 0.85 factor in $\beta = 3/(F \cdot 0.85 \cdot R_p)$ is a temporary debug scaling factor (reduced from 1.0, tried 0.80 first) that shrinks the effective particle radius to increase reaction rates. It was used solely to confirm that the branch A traffic jam (Challenge 4) is rate-limited rather than thermodynamic. The nominal Prada 2013 radius $R_p = 50$ nm should be restored once the traffic jam is resolved by other means.

3. **Entropic contribution**: $dU/dT = 0$ in the OCP model (thermal effects on OCP not implemented).

4. **Singular OCP at extreme θ**: No guard against $\ln(\theta / (1 - \theta)) \to \pm\infty$ as $\theta \to 0$ or $\theta \to 1$.

5. **Initial condition accuracy**: Gamma(2, λ) with two-pass correction still has ~0.3% error on $\theta_0$. A pre-computed steady-state PDF would eliminate this (described in `FUTURE_REFINEMENTS.md`).

6. **Three-subdomain migration**: The sibling worktree `PyBaMM_acm` uses a cleaner three-subdomain architecture (no masks, natural FV boundary handling, 521 total points). The single-domain approach in this worktree was a prototype; migrating to three subdomains would resolve the masking artifacts and potentially the traffic jam issue.

7. **Particle size distribution (PSD)**: Currently single-particle model. Extending to a distribution of particle sizes with per-size Ω would capture heterogeneous phase separation kinetics.
