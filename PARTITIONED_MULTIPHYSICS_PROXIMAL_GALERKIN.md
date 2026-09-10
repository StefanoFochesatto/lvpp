# Partitioned Multiphysics Simulations via Proximal Galerkin: Literature Review & Architectural Framework

**Author / Context:** Research initiative extending `firedrake/lvpp` (Latent Variable Proximal Point / Proximal Galerkin) to partitioned multiphysics simulations (Fluid-Structure-Contact Interaction, Poromechanics, Fracture, and Quasi-Variational Inequalities).  
**Date:** September 2026  

---

## 1. Executive Summary & Literature Landscape

### A. The Literature Landscape
A systematic search across the computational mechanics, optimization, and multiphysics coupling literature reveals an immediate fact:

> **No published work currently formulates or implements "Proximal Galerkin" (PG) or "Latent Variable Proximal Point" (LVPP) methods within a partitioned multiphysics coupling framework.**

The scientific landscape is strictly divided between two mature but isolated communities:

1. **The Proximal Galerkin / LVPP Community** (*Keith & Surowiec 2024; Dokken, Farrell, Keith, Papadopoulos, Surowiec 2025; Papadopoulos 2026*):
   - Focuses on infinite-dimensional variational inequalities (VIs) transformed into smooth saddle-point PDEs via Legendre conjugates ($\nabla R^*$).
   - Numerical implementations to date are **monolithic**: single-domain obstacle problems, single-physics contact mechanics (Signorini), phase-field fracture, liquid crystals, and Monge–Ampère.
   - Where multiphysics has been touched upon (e.g. Cahn–Hilliard multiphase flow in `LVPP.pdf` §3.4, thermoforming QVI in `LVPP.pdf` §3.5 and `hierarchicalPG.pdf` §6.4), the implementation was either solved **monolithically** or used to demonstrate that naive, unaccelerated partitioned fixed-point iterations stall (requiring 164 iterations and 8,493 linear solves vs. 13 iterations for LVPP).

2. **The Partitioned Multiphysics Community** (*preCICE, FSI with contact, poromechanics, coupled fracture, domain decomposition*):
   - Relies on modular coupling of distinct single-physics codes (e.g., OpenFOAM, CalculiX, Firedrake, SU2) communicating via surface coupling libraries.
   - Standard acceleration schemes (IQN-ILS, Anderson acceleration, Robin–Robin transmission) are mathematically grounded in multi-secant quasi-Newton methods assuming **smooth ($C^1$) interface response operators**.
   - When inequality constraints (unilateral contact, cavitation, non-negative saturation, valve diodes) enter the interface or subdomains, standard partitioned methods suffer severe stability and convergence breakdowns.

---

## 2. The Fundamental Bottlenecks in Partitioned Multiphysics with Inequalities

In partitioned simulations, inequality constraints arise either:
- **Subdomain-internally**: crack irreversibility ($c \ge c_{\text{prev}}$), saturation bounds ($0 \le S_w \le 1$), yield surfaces in plasticity.
- **On coupling interfaces**: non-penetration contact ($\boldsymbol{u} \cdot \boldsymbol{n} \le g$, $\lambda \ge 0$), valve/diode flow conditions, cavitation pressure floors ($p \ge p_{\text{cav}}$).

Existing partitioned methods handle these constraints via penalty methods, active-set / semismooth Newton, or augmented Lagrangians. Each introduces well-documented failure modes:

| Method in Partitioned Coupling | Coupling Mechanism | Failure Mode in Partitioned Schemes |
| :--- | :--- | :--- |
| **Penalty Methods** | Regularize inequalities via artificial stiffness $k_\epsilon [g - u]_+$ | Extreme stiffness exacerbates the **added-mass instability** in FSI; requires tiny time steps or causes explosive divergence. |
| **Active-Set / Semismooth Newton** | Sharp combinatorial switching between active/inactive sets | The Dirichlet-to-Neumann (DtN) interface operator becomes **non-smooth ($C^0$, piecewise differentiable)**. Multi-secant accelerators (IQN-ILS, Anderson acceleration) suffer from **interface chattering**, loss of superlinear convergence, and line-search stagnation. |
| **Augmented Lagrangian** | Nested multiplier updates on the interface | Requires **nested iteration loops** (inner contact loop inside outer partitioned FSI loop), multiplying computational expense by an order of magnitude. |
| **Infeasible Intermediate Iterates** | Standard solvers generate unconstrained intermediate updates | Intermediate partitioned iterates violate physical bounds (e.g., negative fluid densities or pressures), causing black-box participant solvers to crash before convergence. |

---

## 3. Four Architectural Paradigms for Partitioned Proximal Galerkin

The Proximal Galerkin / LVPP framework offers four mathematically rigorous pathways to resolve these bottlenecks:

```
                            PARTITIONED PROXIMAL GALERKIN
                                         │
     ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
     ▼                   ▼                               ▼                   ▼
[Paradigm 1]        [Paradigm 2]                    [Paradigm 3]        [Paradigm 4]
Subdomain PG        Interface LVPP                  QVI Proximal        Bregman-ADMM
Smooth Participant  Smooth Robin Transmission       Fixed-Point Loop    Domain Decomposition
(preCICE-ready)     C1 Contact Boundary             QVI Acceleration    Parallel Subdomains
```

---

### Paradigm 1: Subdomain-Level Proximal Galerkin as a $C^1$-Smooth Participant (preCICE-Ready)

#### Concept
Keep the partitioned architecture intact (e.g., using preCICE with standard IQN-ILS / Anderson acceleration). If one participant contains an inequality constraint (e.g., an elastic body undergoing internal contact or phase-field fracture), that participant solves its problem using **Proximal Galerkin** rather than an active-set or penalty solver.

#### The Mathematical Mechanism
Instead of solving a nonsmooth variational inequality, the participant solves a sequence of **smooth, continuously Fréchet-differentiable PDEs**:
$$\alpha_k \langle J'(u^k), v \rangle + (\psi^k, v) = \alpha_k \langle f, v \rangle + (\psi^{k-1}, v)$$
$$(u^k, w) - (\nabla R^*(\psi^k), w) = 0$$

#### Impact on the Partitioned Coupler
Because $\nabla R^*$ (e.g. the $\tanh$ or exponential map) is $C^\infty$-smooth, the interface Dirichlet-to-Neumann response map:
$$\mathcal{S}_{\Gamma}: u_\Gamma \mapsto \sigma(u) \cdot \boldsymbol{n}|_\Gamma$$
is **$C^1$-differentiable**. 

#### Advantage
The interface Jacobian $D\mathcal{S}_\Gamma$ is continuous. Black-box quasi-Newton methods (IQN-ILS) no longer encounter discontinuous derivative jumps across active-set transitions, restoring theoretical superlinear convergence without modifying the coupling library.

---

### Paradigm 2: The Interface LVPP / Proximal Robin Transmission Condition

#### Concept
For inequalities situated directly **on the coupling interface** (e.g., a flexible valve slamming against a rigid wall, or contact between two submerged structures).

#### The Classical Problem
On the contact interface $\Gamma_C$, the Signorini conditions are non-smooth:
$$u_1 \cdot n_1 - u_2 \cdot n_2 \le g, \quad \lambda_N \ge 0, \quad \lambda_N (g - (u_1 \cdot n_1 - u_2 \cdot n_2)) = 0$$

#### The Proximal Robin Formulation
Introduce an **interface latent variable $\psi_\Gamma \in L^\infty(\Gamma_C)$**. By Table 1 of `LVPP.pdf`, using the unilateral upper-bound Shannon entropy:
$$u_1 \cdot n_1 - u_2 \cdot n_2 = g - \exp(-\psi_\Gamma) = \nabla R^*(\psi_\Gamma)$$

The transmission conditions between Subdomain 1 (fluid/solid) and Subdomain 2 (structure/obstacle) transform into a smooth **Proximal Robin–Neumann pair**:
$$\begin{aligned}
\textbf{Subdomain 1 (Robin):} \quad &\sigma_1 \boldsymbol{n}_1 + \psi_\Gamma \boldsymbol{n}_1 = 0 \quad \text{on } \Gamma_C \\
\textbf{Subdomain 2 (State/Primal):} \quad &(u_2 \cdot \boldsymbol{n}_2, w)_{\Gamma_C} + (\exp(-\psi_\Gamma), w)_{\Gamma_C} = (u_1 \cdot \boldsymbol{n}_1 - g, w)_{\Gamma_C}
\end{aligned}$$

#### Advantage
1. The interface condition is completely smooth for any finite $\psi_\Gamma$.
2. At each partitioned iteration, Subdomain 1 receives a well-behaved, smooth traction $\psi_\Gamma \boldsymbol{n}_1$.
3. The contact pressure $\lambda_N = (\psi_\Gamma - \psi_\Gamma^{\text{prev}})/\alpha_k \ge 0$ is recovered naturally via the proximal drift, without requiring Lagrange multiplier projection spaces or penalty springs.

---

### Paradigm 3: Outer Proximal Point Acceleration for Quasi-Variational Inequalities (QVIs)

#### Concept
Coupled multiphysics where the constraint in one field depends directly on the state of the other field (e.g. thermo-mechanical contact, where thermal expansion deforms the contact obstacle: $\Phi = \Phi(T)$).

#### Evidence from the Literature
In `LVPP.pdf` (§3.5, Table 2) and `hierarchicalPG.pdf` (§6.4), the authors compared solvers on the coupled thermoforming QVI:
- Standard partitioned fixed-point iteration (freeze $u$, solve thermal PDE; freeze $T$, solve mechanical VI): **required 164 outer iterations and 8,493 linear solves** because the fixed-point contraction factor was close to 1.
- Monolithic LVPP: **converged in 13 iterations and 20 linear solves**.

#### Partitioned Proximal Acceleration
Instead of unaccelerated Gauss–Seidel fixed-point coupling, wrap the partitioned solver inside an **outer proximal point iteration**:
$$z^{k} = \operatorname{argmin}_{z} \left[ \mathcal{J}_{\text{coupled}}(z) + \frac{1}{\alpha_k} D_R(B z, B z^{k-1}) \right]$$
By Theorem 4.13 in `ProximalGalerkin.pdf`, the proximal point operator regularizes the spectrum of the coupled operator, accelerating contractive fixed-point maps from sublinear / stagnant rates to superlinear convergence.

---

### Paradigm 4: Bregman-ADMM and Proximal Schwarz Decomposition

#### Concept
For large-scale parallel multiphysics across non-overlapping subdomains $\Omega_1$ and $\Omega_2$ with interface $\Gamma$.

#### Formulation
Formulate the coupled problem as constrained optimization with split variables $u_1 \in V_1, u_2 \in V_2$ subject to $u_1|_\Gamma = u_2|_\Gamma$ and local physical bounds $u_i \in K_i$.

#### Algorithm
Apply the Alternating Direction Method of Multipliers (ADMM) where the augmented Lagrangian penalty is replaced by a **Bregman divergence** $D_R$:
$$\begin{aligned}
u_1^{k+1} &= \operatorname{argmin}_{u_1 \in K_1} \left[ J_1(u_1) + \langle \lambda^k, u_1 \rangle + \frac{1}{\alpha} D_{R_1}(u_1, u_1^k) \right] \\
u_2^{k+1} &= \operatorname{argmin}_{u_2 \in K_2} \left[ J_2(u_2) + \langle \lambda^k, -u_2 \rangle + \frac{1}{\alpha} D_{R_2}(u_2, u_2^k) \right] \\
\lambda^{k+1} &= \lambda^k + \rho (u_1^{k+1}|_\Gamma - u_2^{k+1}|_\Gamma)
\end{aligned}$$

#### Advantage
Each subdomain solve is an independent Proximal Galerkin subproblem. Parallel execution is preserved, and the Bregman metric enforces strict feasibility ($u_i \in \text{int } K_i$) within each single-physics solver throughout the partitioned iterations.

---

## 4. Prime Multiphysics Target Benchmarks

The following coupled physical systems would benefit most from a partitioned Proximal Galerkin implementation:

1. **Fluid-Structure Interaction with Impact / Valve Closure (FSCI)**:
   - *Physics*: Incompressible Navier–Stokes coupled with an elastic leaflet impacting a rigid wall or adjacent leaflet.
   - *Current limitation*: Partitioned FSI solvers (e.g. preCICE + OpenFOAM + CalculiX) fail when leaflets touch because the fluid mesh collapses and the structural active set switches discontinuously.
   - *PG advantage*: The contact gap $g - \boldsymbol{u} \cdot \boldsymbol{n} = \exp(-\psi_\Gamma)$ remains strictly positive ($\exp(-\psi) > 0$ for all finite $\psi$). The fluid mesh never undergoes zero-thickness element inversion, and the structural contact force is smooth.

2. **Poroelasticity with Cavitation and Saturation Limits**:
   - *Physics*: Biot consolidation coupled with two-phase flow in porous media. Fluid pressures are bounded below by cavitation: $p \ge p_{\text{cav}}$, and water saturation is bounded: $0 \le S_w \le 1$.
   - *Current limitation*: Standard partitioned drained/undrained splits allow intermediate pressure iterates to drop below cavitation, crashing the fluid equation of state.
   - *PG advantage*: The Fermi–Dirac or Shannon reconstruction guarantees $p \in \text{int } C$ and $S_w \in (0, 1)$ pointwise at every sub-iteration.

3. **Fluid-Driven Phase-Field Fracture (Hydraulic Fracturing)**:
   - *Physics*: Stokes / Darcy flow inside a propagating crack coupled to elastic deformation and phase-field fracture damage $c \in [c_{\text{prev}}, 1]$.
   - *Current limitation*: Partitioned schemes alternate between displacement $u$, fluid pressure $p_f$, and crack phase field $c$. Crack irreversibility ($c \ge c_{\text{prev}}$) causes active-set oscillation that destabilizes pressure coupling.
   - *PG advantage*: Phase field is updated via Fermi–Dirac LVPP (§3.3 in `LVPP.pdf`), ensuring $C^1$-smooth traction updates to the fluid solver.

---

## 5. Research & Implementation Roadmap

```
Phase 1: Participant-Level Validation (Months 1–3)
  ├── Build preCICE participant adapter for firedrake/lvpp
  ├── Benchmark against thermoforming QVI with standard IQN-ILS acceleration
  └── Quantify quasi-Newton convergence gain vs active-set / penalty participants

Phase 2: Interface Proximal Robin Transmission (Months 4–6)
  ├── Implement interface latent field psi_Gamma on coupling meshes
  ├── Formulate Proximal Robin-Neumann transmission for contact-FSI
  └── Demonstrate mesh-independent partitioned iteration count without chattering

Phase 3: Time-Stepping vs Proximal Parameter Synchronization (Months 7–9)
  ├── Analyze interaction between physical time step Δt and proximal parameter α_k
  ├── Determine criteria for single-proximal-step-per-time-step (implicit-explicit PG)
  └── Prove discrete energy stability at the coupled interface
```

---

## 6. References & Key Citations

1. **B. Keith & T. M. Surowiec**, *Proximal Galerkin: A structure-preserving finite element method for pointwise bound constraints*, Foundations of Computational Mathematics (2024).
2. **J. S. Dokken, P. E. Farrell, B. Keith, I. P. A. Papadopoulos, & T. M. Surowiec**, *The latent variable proximal point algorithm for variational problems with inequality constraints*, arXiv:2503.05672 (2025).
3. **I. P. A. Papadopoulos**, *Hierarchical proximal Galerkin: a fast hp-FEM solver for variational problems with pointwise inequality constraints*, arXiv:2412.13733 (2026).
4. **B. Uekermann et al.**, *preCICE: Software for partitioned multi-physics simulations*, ACM Transactions on Mathematical Software (2021).
5. **H. F. Walker & P. Ni**, *Anderson acceleration for fixed-point iterations*, SIAM J. Numer. Anal., 49(4), 1715–1735 (2011).
6. **C. Chniti et al.**, *Robin-Robin domain decomposition methods for non-smooth contact mechanics*, Comput. Methods Appl. Mech. Engrg. (2017).
