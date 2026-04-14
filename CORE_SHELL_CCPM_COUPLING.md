# Core-Shell + CCPM Coupling: Formal Specification

## 1. Physical Picture

Each LFP particle in branch B of the CCPM framework has a core-shell internal structure:

- **Core** (0 ≤ r ≤ r_p): uniform α-phase at concentration c_sp1
- **Shell** (r_p ≤ r ≤ R_p): β-phase with a radial concentration profile c(r,t)
- **Phase boundary** at r = r_p(t): moves as α converts to β (lithiation) or β to α (delithiation)

The **average concentration** of the whole particle:

$$c_{\text{avg}} = \frac{3}{R_p^3}\left[\int_0^{r_p} c_{\text{sp1}}\, r^2\, dr + \int_{r_p}^{R_p} c(r,t)\, r^2\, dr\right] = \hat{r}^3\, c_{\text{sp1}} + (1 - \hat{r}^3)\, \bar{c}_{\text{shell}}$$

where $\hat{r} = r_p / R_p$ is the nondimensional core radius.

---

## 2. CCPM Framework

In CCPM, particles are described by a probability density $g_b(c_p, t)$ on a fixed concentration grid $c_p \in [c_1^*, c_2^*]$. Each point $c_p$ represents the population of particles whose **average concentration equals $c_p$**.

$c_p$ is a state label, not a spatial coordinate. A particle at position $c_p$ on this grid has $c_{\text{avg}} = c_p$ by definition.

---

## 3. Internal State at Each c_p Bin

A particle at grid position $c_p$ carries internal degrees of freedom:

$$\text{Internal state:} \quad \{\hat{r},\, \hat{c}_1, \hat{c}_2, \ldots, \hat{c}_N\}$$

- $\hat{r}$: nondimensional phase-front radius
- $\hat{c}_j = c(\chi_j) / c_{\max}$: nondimensional shell concentration at Landau node $\chi_j = j/N$

### Consistency constraint (always holds)

$$c_p = \hat{r}^3\, c_{\text{sp1}} + (1 - \hat{r}^3)\, \bar{c}_{\text{shell}}(\hat{r},\, \{\hat{c}_j\})$$

where

$$\bar{c}_{\text{shell}} = \frac{3\, c_{\max}\, \hat{\ell}}{1 - \hat{r}^3} \sum_{j=1}^{N} \hat{c}_j \cdot (\hat{r} + j\Delta\chi\, \hat{\ell})^2\, \Delta\chi$$

This constraint is **not imposed separately** — it is **guaranteed by construction** when the evolution equations are correct (see §5 mass conservation proof).

---

## 4. Landau Coordinate Transform (Fixed Mesh for Moving Boundary)

Following Zhuo et al. (2023), the physical shell domain $[r_p, R_p]$ is mapped to a fixed computational domain $\chi \in [0, 1]$:

$$r = R_p(\hat{r} + \chi\, \hat{\ell}), \qquad \hat{\ell} = 1 - \hat{r}$$

- $\chi = 0$: core-shell interface ($r = r_p$)
- $\chi = 1$: particle surface ($r = R_p$)

This avoids remeshing as $r_p$ moves. The price is a **Landau convection term** in the PDE: when $\hat{r}$ changes, a fixed $\chi_j$ node corresponds to a shifting physical position, creating apparent advection.

---

## 5. Lagrangian Evolution (Following One Particle)

In the frame moving with the particle along $c_p$:

### 5a. Shell diffusion (at each Landau node j = 1,...,N)

$$\frac{D\hat{c}_j}{Dt} = \mathcal{D}_j + \mathcal{C}_j$$

**Diffusion operator** (spherical, in Landau coordinates):

$$\mathcal{D}_j = \frac{\hat{D}}{\hat{\ell}^2}\left(\frac{\hat{c}_{j+1} - 2\hat{c}_j + \hat{c}_{j-1}}{\Delta\chi^2} + \frac{2\hat{\ell}}{\hat{r}_j}\cdot\frac{\hat{c}_{j+1} - \hat{c}_{j-1}}{2\Delta\chi}\right)$$

**Landau convection** (mesh motion correction due to $\hat{r}$ changing):

$$\mathcal{C}_j = \frac{(1-\chi_j)\,\dot{\hat{r}}}{\hat{\ell}} \cdot \frac{\hat{c}_{j+1} - \hat{c}_{j-1}}{2\Delta\chi}$$

where:
- $\hat{D} = D_s / R_p^2$ (characteristic diffusion rate)
- $\hat{r}_j = \hat{r} + j\Delta\chi\, \hat{\ell}$ (nondim radial position of node j)
- $\dot{\hat{r}} = D\hat{r}/Dt$ (Lagrangian rate of phase-front motion)

### 5b. Stefan condition (phase-front ODE)

$$\frac{D\hat{r}}{Dt} = \frac{-\hat{D}\, c_{\max}\, (\hat{c}_1 - \theta_2^*)}{(c_2^* - c_{\text{sp1}})\, \hat{\ell}\, \Delta\chi}$$

**Sign convention**: During lithiation (discharge), Li diffuses inward through the shell → $\hat{c}_1 > \theta_2^*$ → $D\hat{r}/Dt < 0$ → core shrinks. During delithiation (charge), the reverse.

### 5c. Boundary conditions in χ

- **Inner** ($\chi = 0$, core-shell interface): Dirichlet

$$\hat{c}_0 = \theta_2^* = c_2^* / c_{\max}$$

- **Outer** ($\chi = 1$, particle surface): Neumann (electrochemical flux)

$$\hat{c}_{N+1} = \hat{c}_{N-1} - \frac{2\Delta\chi\, \hat{\ell}\, R_p\, j_{\text{tr,b}}}{F\, D_s\, c_{\max}}$$

Sign convention: positive $j_{\text{tr,b}}$ = anodic (delithiation, Li leaving particle).

---

## 6. Mass Conservation Proof

Total Li in the particle:

$$N_{\text{Li}} = \frac{4\pi R_p^3}{3}\, c_{\text{avg}} = \frac{4\pi R_p^3}{3}\, c_p$$

The Stefan condition is derived from mass balance at the phase front:

$$\text{Li flux into interface from shell} = \text{Li consumed by phase conversion}$$

$$-D_s \left.\frac{\partial c}{\partial r}\right|_{r_p^+} = (c_2^* - c_{\text{sp1}})\, \dot{r}_p$$

Combined with the shell diffusion equation and BCs, total particle mass conservation gives:

$$\frac{DN_{\text{Li}}}{Dt} = 4\pi R_p^2 \left(-\frac{j_{\text{tr,b}}}{F}\right)$$

Therefore:

$$\frac{Dc_{\text{avg}}}{Dt} = \frac{Dc_p}{Dt} = -\frac{3\, j_{\text{tr,b}}}{F\, R_p\, c_{\max}} \equiv R_b$$

**The rate of change of c_avg equals R_b.** The consistency constraint ($c_{\text{avg}} = c_p$) is maintained automatically by the coupled Stefan + shell diffusion + surface BC system.

---

## 7. Eulerian Form on the Fixed c_p Grid

The material derivative relates Lagrangian and Eulerian frames:

$$\frac{D(\cdot)}{Dt} = \frac{\partial(\cdot)}{\partial t} + R_b\, \frac{\partial(\cdot)}{\partial c_p}$$

### 7a. PDF (conservative advection + sources)

$$\frac{\partial g_b}{\partial t} + \frac{\partial}{\partial c_p}(R_b\, g_b) = S_b$$

### 7b. Shell concentrations (material advection + shell physics)

$$\frac{\partial \hat{c}_j}{\partial t} + R_b\, \frac{\partial \hat{c}_j}{\partial c_p} = \mathcal{D}_j + \mathcal{C}_j$$

Note: **non-conservative** (material) advection. $\hat{c}_j$ is an intensive property carried by the particle, not a conserved density.

### 7c. Phase front (material advection + Stefan)

$$\frac{\partial \hat{r}}{\partial t} + R_b\, \frac{\partial \hat{r}}{\partial c_p} = \frac{-\hat{D}\, c_{\max}\, (\hat{c}_1 - \theta_2^*)}{(c_2^* - c_{\text{sp1}})\, \hat{\ell}\, \Delta\chi}$$

---

## 8. Why Both c_p Advection and Landau Convection Are Needed

$\hat{c}_j$ has **two separate advection mechanisms**:

1. **c_p advection** ($R_b\, \partial \hat{c}_j / \partial c_p$): The particle moves along the $c_p$ grid because its average concentration changes. This carries the internal shell state with the particle population. Without this term, the shell state at a $c_p$ bin would be "inherited" from whatever particle previously occupied that bin — incorrect under transient conditions.

2. **Landau convection** ($\mathcal{C}_j$): The phase boundary $r_p$ moves inside the particle, so a fixed $\chi_j$ node corresponds to a shifting physical location. This is a coordinate-mapping artifact, not physical transport. It vanishes when $\dot{\hat{r}} = 0$ (stationary phase front).

These operate on **different axes** ($c_p$ vs. $\chi$) and represent different physics (macro-transport vs. micro-geometry).

---

## 9. Interface Kinetics

### Surface concentration

$$c_{\text{surf}} = c_{\max}\, \hat{c}_N$$

$\theta_{\text{surf}} = \hat{c}_N$ (outermost shell node).

### Branch B Butler-Volmer

$$j_{\text{tr,b}} = j_0'\, \sqrt{\frac{c_e}{c_{e,0}}}\, \sqrt{\theta_{\text{surf}}(1 - \theta_{\text{surf}})}\, \cdot 2\sinh\!\left(\frac{F\eta}{2RT}\right)$$

Without core-shell: $\theta_{\text{surf}}$ is a constant ($\theta_1^*$ or $\theta_2^*$).
With core-shell: $\theta_{\text{surf}}$ is **dynamic**, from the shell PDE solution.

### Advection velocity

$$R_b(c_p) = -\frac{3\, j_{\text{tr,b}}(c_p)}{F\, R_p\, c_{\max}}$$

### Total current (coupling to DFN)

$$j_{\text{tot}} = \int_{c_1^*}^{c_2^*} g_b(c_p)\, j_{\text{tr,b}}(c_p)\, dc_p + \text{(branch A and C contributions)}$$

---

## 10. Initial / Boundary Conditions on c_p Grid

### At c_p = c_sp1 (left edge of branch B)

Mass entering from branch A arrives with particles that have just nucleated a shell:

- $\hat{r} = 1 - \epsilon$ (vanishingly thin shell)
- $\hat{c}_j = \theta_2^*$ for all j (shell at equilibrium)
- $c_{\text{surf}} = c_2^*$

These serve as the **upwind boundary values** for the material advection of $\hat{r}$ and $\hat{c}_j$ into branch B during discharge (lithiation, $R_b > 0$).

### At c_p = c_2* (right edge of branch B)

- $\hat{r} \to 0$ (core has vanished, particle is fully β-phase)
- Mass exits to branch C via existing B→C transition flux
- No upstream BC needed (outflow boundary for $R_b > 0$)

### Where g_b = 0

Shell states are meaningless. All RHS terms multiplied by a smooth gate:

$$\text{gate} = \sigma(g_b - \epsilon, k)$$

States freeze at ICs. Zero contribution to $j_{\text{tot}}$ because $g_b = 0$.

---

## 11. FV Discretization of Material Advection

The material advection $R_b\, \partial \hat{c}_j / \partial c_p$ uses **upwind differencing** (same velocity field as $g_b$):

$$R_b\, \frac{\partial \hat{c}_j}{\partial c_p}\bigg|_k \approx R_b^+\, \frac{\hat{c}_j^{(k)} - \hat{c}_j^{(k-1)}}{\Delta c_p} + R_b^-\, \frac{\hat{c}_j^{(k+1)} - \hat{c}_j^{(k)}}{\Delta c_p}$$

where $R_b^+ = \max(R_b, 0)$, $R_b^- = \min(R_b, 0)$, and $k$ is the $c_p$ bin index.

Note the **difference** from $g_b$ advection:
- $g_b$: **conservative** form $\partial(R_b g_b)/\partial c_p$ — preserves total mass
- $\hat{c}_j$, $\hat{r}$: **material** (non-conservative) form $R_b\, \partial(\cdot)/\partial c_p$ — carries intensive quantities

---

## 12. State Count and Computational Cost

| Variable | Type | Count per x-node | Active fraction |
|---|---|---|---|
| $g_b(c_p)$ | ODE, conservative advection | $N_{cp,B}$ | ~15% |
| $\hat{c}_j(c_p)$, j=1,...,N | ODE, material advection + diffusion | $N \times N_{cp,B}$ | ~15% |
| $\hat{r}(c_p)$ | ODE, material advection + Stefan | $N_{cp,B}$ | ~15% |

With N_shell = 3, N_{cp,B} = 242, N_x = 20:

- Formal new states: $(3 + 1) \times 242 \times 20 = 19{,}360$
- Active states (where $g_b > 0$): ~15% ≈ 2,900
- Sparse KLU solver exploits zero Jacobian rows for inactive states

---

## 13. Summary of Equations

| Equation | Form | Physics |
|---|---|---|
| $\partial_t g_b + \partial_{c_p}(R_b g_b) = S_b$ | Conservative | PDF population transport |
| $\partial_t \hat{c}_j + R_b \partial_{c_p} \hat{c}_j = \mathcal{D}_j + \mathcal{C}_j$ | Material | Shell Li diffusion + Landau correction |
| $\partial_t \hat{r} + R_b \partial_{c_p} \hat{r} = -\hat{D} c_{\max} (\hat{c}_1 - \theta_2^*) / [(c_2^* - c_{\text{sp1}}) \hat{\ell} \Delta\chi]$ | Material | Stefan phase-front motion |

Feedback loop:

$$j_{\text{tr,b}} \xrightarrow{\text{surface BC}} \hat{c}_j \text{ evolve} \xrightarrow{c_{\text{surf}}} j_{\text{tr,b}} \qquad \text{(implicit coupling)}$$

$$\hat{c}_j \xrightarrow{\text{interface gradient}} \dot{\hat{r}} \xrightarrow{\text{geometry}} \mathcal{D}_j,\, \mathcal{C}_j \xrightarrow{\text{diffusion}} \hat{c}_j$$
