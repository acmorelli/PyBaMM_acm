# Core-Shell CCPM: Current Implementation (as of 2026-04-14)

This document describes the **current** state of the core-shell solid diffusion model
coupled to the CCPM framework, as implemented in `ccpm_snap_mesh_Ra_smooth`.

---

## 1. Overview

The core-shell model adds intra-particle phase-transition diffusion to branch B
of the CCPM. Each $c_p$ bin in branch B carries an independent radial shell problem:
a moving phase boundary ($\hat{r}$) and a shell concentration profile ($\hat{c}_j$).

**Key files:**

| File | Role |
|---|---|
| `src/.../particle/ccpm_positive_particle.py` | Shell variables, PDEs, ICs |
| `src/.../interface/kinetics/ccpm_positive_interface.py` | $j_{\text{tr,b}}$ using $c_{\text{surf}}$ |
| `src/.../full_battery_models/lithium_ion/dfn_ccpm.py` | Model assembly |

---

## 2. State Variables

All live on the `cp_domains` = {primary: `"CCPM positive particle concentration"`,
secondary: `"positive electrode"`, tertiary: `"current collector"`}.

| Variable | Symbol | Nondim | Meaning |
|---|---|---|---|
| `"Shell phase front r_hat"` | $\hat{r}$ | $r_p / R_p$ | Core radius / particle radius |
| `"Shell concentration hat chi_j"` ($j=1..N$) | $\hat{c}_j$ | $c_j / c_{\max}$ | Shell concentration at Landau node $j$ |

$N = N_{\text{shell}} = 3$ (default). $\Delta\chi = 1/N = 1/3$.

Dimensional outputs for diagnostics:
- `"Shell phase front r_p"` $= R_p \cdot \hat{r}$
- `"Shell concentration chi_j"` $= c_{\max} \cdot \hat{c}_j$

---

## 3. Coordinate Transform (Landau)

Physical shell domain $[r_p, R_p]$ mapped to fixed $\chi \in [0, 1]$:

$$r = R_p(\hat{r} + \chi\,\hat{\ell}), \qquad \hat{\ell} = 1 - \hat{r}$$

- $\chi = 0$: core-shell interface ($r = r_p$)
- $\chi = 1$: particle surface ($r = R_p$)
- Node positions: $\hat{r}_j = \hat{r}_{\text{safe}} + j\Delta\chi\,\hat{\ell}$

---

## 4. Clamping of r_hat

```python
r_hat_safe = max(min(shell_r_hat, 0.99), 0.01)
ell_hat = 1.0 - r_hat_safe
```

Applied to **geometry computations only** (computing $\hat{\ell}$, $\hat{r}_j$, diffusion/convection).
The ODE state `shell_r_hat` itself is not clamped.

Additionally, **smooth sigmoid gates** on the RHS prevent $\hat{r}$ from leaving $[0, 1]$:

```python
upper_gate = 1 - sigmoid(r_hat, 0.98, 200)   # → 0 as r_hat → 1
lower_gate = sigmoid(r_hat, 0.02, 200)        # → 0 as r_hat → 0
drhat_dt = max(drhat_dt, 0) * upper_gate + min(drhat_dt, 0) * lower_gate
```

---

## 5. Diffusivity

Evaluated once at $\theta = c_2^* / c_{\max}$ (β-phase binodal) and current temperature.
**Constant within the shell** — no composition dependence.

$$\hat{D} = D_s / R_p^2$$

```python
D_s = self.param.p.prim.D(c2_star_val / c_p_max_trunc, T_p_bc)
D_hat = D_s / R_p**2
```

---

## 6. Boundary Conditions in χ

### Inner (χ = 0, core-shell interface): Dirichlet

$$\hat{c}_0 = \theta_2^* = c_2^* / c_{\max}$$

### Outer (χ = 1, particle surface): Neumann via ghost node

$$\hat{c}_{N+1} = \hat{c}_{N-1} - \frac{2\Delta\chi\,\hat{\ell}\,R_p\,j_{\text{tr,b}}}{F\,D_s\,c_{\max}}$$

Derived from:

$$\frac{D_s\,c_{\max}}{R_p\,\hat{\ell}} \cdot \frac{\hat{c}_{N+1} - \hat{c}_{N-1}}{2\Delta\chi} = -\frac{j_{\text{tr,b}}}{F}$$

Sign: positive $j_{\text{tr,b}}$ = delithiation (Li leaving) → ghost node lower than $\hat{c}_{N-1}$.

---

## 7. Stefan Condition (r_hat ODE)

$$\frac{\partial \hat{r}}{\partial t} = \frac{-\hat{D}\,c_{\max}\,(\hat{c}_1 - \theta_2^*)}{(c_2^* - c_{\text{sp1}})\,\hat{\ell}\,\Delta\chi} \cdot \text{mask\_b}$$

**Sign convention** (corrected 2026-04-14):
- During lithiation: $\hat{c}_1 > \theta_2^*$ → $\partial_t \hat{r} < 0$ → core shrinks ✓
- During delithiation: $\hat{c}_1 < \theta_2^*$ → $\partial_t \hat{r} > 0$ → core grows ✓

Then gated by sigmoid bounds (see §4).

**Note**: This is a **pure Lagrangian** $D\hat{r}/Dt$ applied as a partial derivative
$\partial_t \hat{r}$. There is **no $R_b\,\partial_{c_p}\hat{r}$ advection term** —
the shell state does not travel with the particle along $c_p$.

---

## 8. Shell Concentration PDEs

For each Landau node $j = 1, \ldots, N$ at $\chi_j = j\Delta\chi$:

$$\frac{\partial \hat{c}_j}{\partial t} = \left[\mathcal{D}_j + \mathcal{C}_j\right] \cdot \text{mask\_b}$$

### Diffusion (spherical, Landau coordinates)

$$\mathcal{D}_j = \frac{\hat{D}}{\hat{\ell}^2}\left(\frac{\hat{c}_{j+1} - 2\hat{c}_j + \hat{c}_{j-1}}{\Delta\chi^2} + \frac{2\hat{\ell}}{\hat{r}_{j,\text{safe}}}\cdot\frac{\hat{c}_{j+1} - \hat{c}_{j-1}}{2\Delta\chi}\right)$$

where $\hat{r}_{j,\text{safe}} = \text{smooth\_max}(\hat{r}_j,\, 10^{-6},\, 10^{10})$.

### Landau convection (moving-boundary mesh correction)

$$\mathcal{C}_j = \frac{(1-\chi_j)\,\dot{\hat{r}}}{\hat{\ell}} \cdot \frac{\hat{c}_{j+1} - \hat{c}_{j-1}}{2\Delta\chi}$$

This corrects for the fact that a fixed $\chi_j$ node corresponds to a shifting physical
radius when $\hat{r}$ changes. It is **not** related to $c_p$-axis advection.

**Note**: Like the Stefan condition, there is **no $R_b\,\partial_{c_p}\hat{c}_j$ advection**.
Shell states are tied to their $c_p$ bin, not to any physical particle.

---

## 9. mask_b

Hard binary mask on the $c_p$ grid:

$$\text{mask\_b} = \begin{cases} 1 & c_1^* \le c_p \le c_2^* \\ 0 & \text{otherwise} \end{cases}$$

Zeros all core-shell RHS outside branch B's concentration range.
No smooth transition. No dependence on $g_b$ amplitude.

---

## 10. Interface Kinetics (j_tr_b)

### With core-shell (`solid_diffusion == "core_shell"`)

Surface concentration from outermost shell node:

$$c_{\text{surf}} = c_{\max} \cdot \hat{c}_N$$

Clamped to $(10^{-6} c_{\max},\; c_{\max} - 10^{-6} c_{\max})$ via `smooth_max`/`smooth_min`.

Butler-Volmer (mixed-phase, $\mu_b = 0$):

$$j_{\text{tr,b}} = j_0'\,\sqrt{\frac{c_e}{c_{e,0}}}\,\sqrt{\theta_{\text{surf}}(1-\theta_{\text{surf}})}\,\cdot 2\sinh\!\left(\frac{F\eta}{2RT}\right)$$

### Without core-shell

$$j_{\text{tr,b}} = j_0'\,\sqrt{\frac{c_e}{c_{e,0}}}\,\sqrt{\theta_1^*(1-\theta_1^*)}\,\cdot 2\sinh\!\left(\frac{F\eta}{2RT}\right)$$

Constant prefactor using the Li-poor binodal $\theta_1^*$.

### Total current (DFN coupling)

$$j_{\text{tot}} = \int\left(\text{mask}_a\,g_a\,j_{\text{tr,a}} + \text{mask}_b\,g_b\,j_{\text{tr,b}} + \text{mask}_c\,g_c\,j_{\text{tr,c}}\right) dc_p$$

---

## 11. Advection Velocity R_b

$$R_b = -\frac{3\,j_{\text{tr,b}}}{F\,R_p\,c_{\max}}$$

Note: the code uses $\beta = 3/(F \cdot R_p)$, so $R_b = -\beta \cdot j_{\text{tr,b}}$.
The factor $1/c_{\max}$ is implicitly absorbed because the CCPM uses concentration
(not stoichiometry) on the $c_p$ axis.

---

## 12. Initial Conditions

### r_hat (lever-rule equilibrium)

$$\hat{r}_{\text{init}}(c_p) = \left(\frac{c_2^* - c_p}{c_2^* - c_{\text{sp1}}}\right)^{1/3}$$

Fraction clamped to $[10^{-4},\; 1-10^{-4}]$ before cube root.

- At $c_p = c_{\text{sp1}}$: $\hat{r} \to 1$ (all core, no shell)
- At $c_p = c_2^*$: $\hat{r} \to 0$ (all shell, no core)

### Shell concentrations (uniform equilibrium)

$$\hat{c}_j(t=0) = \theta_2^* \quad \forall j$$

At equilibrium: $\hat{c}_1 = \theta_2^* = \hat{c}_0$ → Stefan gives $\dot{\hat{r}} = 0$.

---

## 13. set_rhs Registration

```python
self.rhs[variables["Shell phase front r_hat"]] = variables["Shell r_hat RHS"]
for j in range(1, self.N_shell + 1):
    self.rhs[variables[f"Shell concentration hat chi_{j}"]] = variables[f"Shell chat_{j} RHS"]
```

---

## 14. Feedback Loop

```
j_tr_b ──→ surface Neumann BC ──→ shell PDE (chat_j evolve)
  ↑                                      │
  │                                      ├──→ c_surf = c_max * chat_N ──→ j_tr_b
  │                                      │
  │                                      └──→ chat_1 ──→ Stefan (dr_hat/dt)
  │                                                        │
  │                                                        └──→ geometry (ell_hat, r_hat_j)
  │                                                               │
  └────────────────────────────────────────────────────────────────┘
```

---

## 15. Known Limitations of Current Approach

### 15a. No c_p advection of internal states

The shell state ($\hat{r}$, $\hat{c}_j$) at each $c_p$ bin is **Eulerian** — it stays at that
bin regardless of mass flow. When $g_b$ advects mass from bin $k$ to $k+1$ at velocity $R_b$,
the arriving particles **inherit** the shell state at bin $k+1$ rather than carrying their own.

**Physical assumption**: the shell profile is a property of the $c_p$ position, not of the particle.

**Valid when**: $\tau_{\text{shell,diff}} \ll \Delta c_p / R_b$ (shell equilibrates faster than one
bin transit). This holds for thin shells (near $c_{\text{sp1}}$) but breaks for thick shells and
high C-rates.

### 15b. Mass consistency not enforced

The constraint $c_{\text{avg}} = \hat{r}^3 c_{\text{sp1}} + (1-\hat{r}^3)\bar{c}_{\text{shell}} = c_p$
is satisfied at $t=0$ (by the lever-rule IC) but can **drift** during evolution because:

- The Stefan condition changes $\hat{r}$ based on the interface gradient
- The shell PDE changes $\bar{c}_{\text{shell}}$ from surface flux
- Neither is constrained to keep $c_{\text{avg}} = c_p$
- This drift manifests as r_hat reaching unphysical values (> 1 or < 0)

### 15c. Stefan sign was wrong (fixed 2026-04-14)

Original code had positive sign → core grew during lithiation. Corrected to negative.

### 15d. r_hat can leave [0, 1]

Despite the sigmoid gates, the ODE state can drift past bounds in bins where $g_b > 0$ under
strong driving. The geometric clamp prevents numerical blowup but doesn't prevent unphysical state.

### 15e. mask_b is a hard step function

Can cause Jacobian discontinuities for implicit solvers. No smooth transition at branch boundaries.

### 15f. No g_b-gating

Shell RHS is active at all $c_p$ bins within $[c_1^*, c_2^*]$, even where $g_b = 0$.
This wastes computation and allows shell states to drift freely in empty bins.

---

## 16. Planned Improvements (see CORE_SHELL_CCPM_COUPLING.md)

1. Add **material advection** $R_b\,\partial_{c_p}$ to $\hat{r}$ and $\hat{c}_j$ equations
2. This makes internal states travel with the particle population
3. Mass conservation ($c_{\text{avg}} = c_p$) is then guaranteed by Stefan + shell PDE + BCs
4. Add **g_b-gating**: multiply all shell RHS by $\sigma(g_b - \epsilon)$ to freeze inactive bins
5. Upwind **inflow BC at $c_{\text{sp1}}$**: $\hat{r} = 1-\epsilon$, $\hat{c}_j = \theta_2^*$
