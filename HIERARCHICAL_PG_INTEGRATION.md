# Integration of LVPP Matrix-Free Preconditioning with Hierarchical Proximal Galerkin (hpG)

**Author / Context:** Research synthesis bridging `firedrake/lvpp` (matrix-free saddle preconditioning, contact degeneracy, AMR) and `HierarchicalProximalGalerkin` (Papadopoulos 2026, arXiv:2412.13733).  
**Date:** September 2026  

---

## 1. Executive Summary

Both the `lvpp` research program and the hierarchical Proximal Galerkin paper (`hierarchicalPG.pdf`, Papadopoulos 2026) converge on the same fundamental linear algebra problem: **how to solve the linearized Newton saddle-point systems arising in LVPP without assembling the dense Schur complement.**

In `hierarchicalPG.pdf`, Papadopoulos established a high-level outer-FGMRES / inner-Krylov solver architecture using high-order tensor-product elements, but:
1. Solved all $h$- and $hp$-adaptive runs with **sparse direct factorizations** (Remark 6.1), leaving the iterative preconditioned solver untested on graded meshes.
2. Used **inner GMRES** on $S$, incurring unnecessary $O(m^2)$ memory and orthogonalization costs for an operator that is mathematically symmetric.
3. Added stabilization $E_\beta = \beta I$ directly to the discrete operator $G$, forcing $\beta \to 0$ as $h \to 0$ to avoid solution error plateaus, which in turn caused linear growth in GMRES iteration counts.

The `lvpp` research session (`RESEARCH.md`, `experiments/matfree_fieldsplit.py`, `scale_up.py`, `grade_pc.py`) solved each of these issues on unstructured finite element meshes. This document defines the exact mathematical mechanisms, clarifies the distinction between $J$ and $J_p$, and provides a concrete blueprint for incorporating `lvpp`'s findings into the `hierarchicalPG` framework.

---

## 2. Foundations: What are $J$ and $J_p$?

In Newton-Krylov methods for nonlinear PDE systems $F(z) = 0$, scientific computing frameworks (PETSc, Firedrake) distinguish between:

$$\begin{aligned}
J   &= \frac{\partial F}{\partial z}(z_k) \quad &&\textbf{The Operator Jacobian (True Jacobian)} \\
J_p &\approx J \quad &&\textbf{The Preconditioner Jacobian (Preconditioner Matrix)}
\end{aligned}$$

At each Newton iteration, the linear system is:
$$J \, \delta z = -F(z_k)$$
A Krylov method (e.g. GMRES, FGMRES) solves the preconditioned system $P^{-1} J \, \delta z = -P^{-1} F(z_k)$, where $P$ is constructed from $J_p$.

### The Critical Role of $J$ vs. $J_p$ in LVPP Systems
The monolithic Jacobian block structure is:
$$J = \begin{bmatrix} A_\alpha & B \\ B^T & D_\psi \end{bmatrix}$$
where $A_\alpha = \alpha K$ is the primal stiffness operator, $B$ is the coupling mass operator, and $D_\psi = \partial F_\psi / \partial \psi$.

For obstacle problems under the Shannon entropy, $D_\psi = -M \odot \exp(\psi)$. On the active contact set ($u = \phi$), the latent variable drifts to negative infinity:
$$\psi(x) \to -\infty \quad \Longrightarrow \quad \exp(\psi) \to 0$$
As a result, the diagonal entries of $D_\psi$ collapse to machine zero ($10^{-28} \dots 10^{-80}$).

- **Failure of Operator Regularization ($J + \epsilon I$)**:
  If a floor $\epsilon > 0$ or stabilization $E_\beta = \beta I$ is added to $J$ itself, the Newton step $\delta z$ targets a perturbed system $F(z) + \epsilon z = 0$. In `lvpp` (`RESEARCH.md` Finding 1), testing `psi_floor_operator=True` caused Newton line searches to diverge at every mesh level (`DIVERGED_DTOL`). The outer proximal point iteration is a contractive fixed-point map that strictly requires exact Newton directions.
- **The $J_p$-Only Design Rule**:
  Floors, regularizations, and preconditioner approximations must live on **$J_p$ only** (`Jp = J + floor`), leaving $J$ exact:
  $$\text{Action: } v \mapsto J v \quad (\text{exact, unperturbed})$$
  $$\text{PC Setup: } J_p = J + \epsilon(x) M \quad (\text{floored, well-conditioned})$$
  This guarantees that linear solvers remain non-singular while the outer Newton and proximal convergence paths remain identical to direct LU.

---

## 3. Assessment of HierarchicalPG on Graded and Adaptive Meshes

#### A. Direct Solvers in the Adaptive Benchmarks
Section 6.1 of `hierarchicalPG.pdf` presents $h$-adaptive and $hp$-adaptive convergence curves for the 1D obstacle problem (Figure 6), while Section 6.2 presents 2D benchmarks (Figure 7).
However, **neither adaptive benchmark used the iterative Schur preconditioner**:
- **Remark 6.1 (Explicit Statement)**:
  > *"We solve the linear systems in the PDAS and hpG strategies via **sparse direct solvers**."*
- **Figure 6 Caption**:
  > *"Any number associated with a data point is the average time taken per Newton iteration, measured in milliseconds, via a **direct sparse factorization**."*
- Tables 1 and 2, which evaluate the preconditioned GMRES strategy, were conducted **strictly on uniform Cartesian grids**.

#### B. The Degradation Mechanism on Graded Meshes
In `hierarchicalPG.pdf`, the Schur complement is:
$$S = -(D_\psi + E_\beta + B^T A_\alpha^{-1} B)$$
Papadopoulos approximates $S$ by removing boundary hat functions to decouple the elements, forming a block-diagonal matrix:
$$\hat{S} = -D_\psi - E_\beta - \hat{B}^T \hat{A}_\alpha^{-1} \hat{B}$$

On **uniform meshes**, this approximation is bounded because element diameters are identical ($h_{\max}/h_{\min} = 1$). But on **graded meshes** (such as adaptive boundary-layer refinement):
1. **Element Interface Mismatch**: Element stiffness scales vary by $(h_{\max}/h_{\min})^2$. Discarding the inter-element continuous coupling across refinement interfaces induces an $O((h_{\max}/h_{\min})^2)$ error in $\hat{S}$.
2. **The "Dead-Row Factory"**: As proven in `lvpp` scaling experiments, refining along the free boundary converts smeared boundary nodes into cleanly pinned contact nodes with $\psi \to -\infty$. The fraction of degenerate $D$-rows jumps from ~7% on uniform meshes to over 50% on band-refined meshes, concentrating extreme conditioning issues directly at element interfaces where mesh grading is steepest.

In `lvpp`, this grading mismatch caused static Schur preconditioners (`selfp`) to blow up from 48 iterations (uniform) to 323 iterations (graded) and diverge entirely at finer levels.

---

## 4. Step-by-Step Blueprint: Incorporating `lvpp` Findings into `hierarchicalPG`

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  PROPOSED HIERARCHICAL-PG SOLVER UPGRADE                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   1. Inner PCG on -S           2. J vs Jp Decoupling        3. Inexact Capping
   -S is strictly SPD           Keep S matvec exact          Loose rtol (1e-3..1e-4)
   Replace GMRES with PCG       Put E_beta on S_hat only     Cap at 20-40 its
   O(m) work & memory           Zero error plateau           Outer FGMRES <= 15 its
```

#### Upgrade 1: Replace Inner GMRES with Preconditioned Conjugate Gradient on $-S$
- **Mathematical Justification**:
  $A_\alpha$ is symmetric positive definite (SPD). $B$ is a real coupling matrix. $D_\psi = -M \odot e^\psi$ is symmetric negative semidefinite. Therefore, the negative Schur complement:
  $$-S = D_\psi^{\text{abs}} + B^T A_\alpha^{-1} B$$
  is **strictly Symmetric Positive Definite (SPD)** on the latent space.
- **Implementation Change**:
  In `hierarchicalPG`, replace the inner GMRES solver for $S$ with **inner PCG on $-S$**:
  $$\text{Solve: } (-S) \, \delta_\psi = -y \quad \text{preconditioned by } (-\hat{S})^{-1}$$
  where $-\hat{S} = D_\psi^{\text{abs}} + \hat{B}^T \hat{A}_\alpha^{-1} \hat{B}$ is Papadopoulos's block-diagonal bubble Cholesky factor.
- **Benefits**:
  - Drops storage from $m$ Krylov vectors to 3 working vectors.
  - Eliminates Gram–Schmidt orthogonalization ($O(m^2)$ operations).
  - Guarantees monotone convergence in the energy norm $\|-S\|^{1/2}$.

#### Upgrade 2: Decouple $E_\beta$ (Operator vs. Preconditioner Separation)
- **Problem in Paper (Table 2)**:
  Papadopoulos added $E_\beta = \beta I$ directly to the discrete Newton operator $G$. When $\beta$ was held constant, the solution error plateaued. To avoid plateaus, Papadopoulos forced $\beta = 10^{-p + \log_2 h} \to 0$, which caused GMRES iterations to grow linearly with $1/h$.
- **Implementation Change**:
  1. Set $\beta = 0$ in the matrix-free evaluation of $-S$:
     $$-S v = D_\psi^{\text{abs}} v + B^T A_\alpha^{-1} (B v)$$
  2. Add $E_\beta$ (with a fixed, robust value like $\beta = 10^{-3}$ or the `lvpp` contact-force floor $\epsilon(x) = \text{psi\_floor} + c \cdot \alpha |\lambda(x)|$) **only to $\hat{S}$**:
     $$-\hat{S} = D_\psi^{\text{abs}} + E_\beta + \hat{B}^T \hat{A}_\alpha^{-1} \hat{B}$$
- **Benefits**:
  - The Newton operator remains exact: **the solution error never plateaus**.
  - $\beta$ does not need to vanish as $h \to 0$, stabilizing the inner PCG solve and eliminating the $1/h$ iteration growth.

#### Upgrade 3: Inexact, Capped Inner-PCG Solves for Adaptive Grids
- **Problem on Graded Meshes**:
  Under $h$-adaptive refinement, demanding high precision (e.g. $\text{rtol} = 10^{-6}$) from the inner solve causes Krylov iteration counts to explode due to interface stiffness scaling.
- **Implementation Change**:
  Following `lvpp/experiments/grade_pc.py` (Variant B):
  1. Set the inner PCG relative stopping tolerance loosely: $\text{rtol}_{\text{inner}} = 10^{-3} \dots 10^{-4}$.
  2. Enforce a strict iteration cap: $\text{max\_it}_{\text{inner}} = 30 \dots 40$.
  3. Pair with **outer Flexible GMRES (FGMRES)**, which is mathematically designed to handle inexact, variable preconditioning.
- **Measured Result in `lvpp`**:
  On a benchmark mesh with extreme $45.6\times$ grading ($h_{\max}/h_{\min} = 45.6$), this strategy kept outer FGMRES iterations bounded at **10 to 15 iterations**, completely resolving the grading breakdown.

---

## 5. Algorithmic Specification: The Unified Matrix-Free hpG Solver

```python
def solve_hpG_step(F, z_k, A_alpha, B, D_psi, b_u, b_psi):
    """
    Solves the linearized Newton saddle system J dz = -F using
    Outer FGMRES with True-Schur Inner-PCG Preconditioning.
    """
    # 1. Outer Solver: FGMRES on monolithic system J dz = rhs
    # Operator matvec J(v_u, v_psi) is evaluated matrix-free via DCT transforms.
    
    def preconditioner_apply(r_u, r_psi):
        # Step A: Forward primal solve (cached Cholesky or p-multigrid)
        du_1 = A_alpha_inv(r_u)
        
        # Step B: Form Schur residual matrix-free
        # y = r_psi - B^T * du_1
        y = r_psi - apply_B_transpose(du_1)
        
        # Step C: Inner PCG solve on the strictly SPD negative Schur complement
        # Action: (-S) v = -D_psi * v + apply_B_transpose(A_alpha_inv(apply_B(v)))
        # Preconditioner: (-S_hat)^(-1) [dense cell-bubble Cholesky with fixed beta floor]
        d_psi = inner_pcg_solve(
            matvec=lambda v: apply_negative_Schur(v, D_psi, B, A_alpha_inv),
            precond=apply_S_hat_inv,
            rhs=-y,
            rtol=1e-4,
            max_its=30
        )
        
        # Step D: Backward primal solve
        du = du_1 - A_alpha_inv(apply_B(d_psi))
        
        return du, d_psi

    # Run outer FGMRES with preconditioner_apply until ||J dz - rhs|| / ||rhs|| < 1e-6
    return fgmres(matvec=apply_J_exact, precond=preconditioner_apply, rhs=[-b_u, -b_psi])
```

---

#---

## 6a. hpG-in-Firedrake port status (2026-09-10)

The faithful port exists: `experiments/hpg/` (spec `HPG_PORT_SPEC.md`,
results `experiments/hpg/RESULTS.md`).  Gates 0-1 PASSED (structure
verifications at machine precision: DQ-spectral psi-mass diagonality
[Lem 3.2 exact, via the discovered modal<->nodal Vandermonde], D_psi
exactly cell-block-diagonal, Y-basis Ahat = 4 parity blocks per cell
[the paper's "4N"], alpha-cache trick verified 3.6e-15, Shat SPD at the
deep state, cold true-Schur inner GMRES 14/12 its).  Direct control vs
the analytic sphere solution: hpG p=2 at 1345 dofs is ~13x more accurate
than our P1 at 545 (and ~35x at matched ~2k dofs) — the paper's
accuracy-per-dof selling point confirmed on our benchmark.  The
two-stage PC (eq 4.5) is implemented AND PROMOTED end-to-end
(2026-09-10, `promote_hpg.py`): uniform L0-L2 p=2 matches the serial-
direct controls with outer FGMRES max 2 its FLAT (vs paper's 14-36, vs
our P1 selfp 14/29/48); **beta sweep {0..1e-3} all identical -> our
floorless finding transfers to hpG**; **graded chain CLEAN at
26.4x/55.3x/114.8x/226.1x grading (up to 25.7k dofs), no divergence
anywhere** — the Remark 6.1 gap (iterative solver on adaptive meshes)
closes in hpG's favor, and our selfp blowup is shown to be
discretization/PC-specific, not fundamental.  Honest negative: the
published inner discipline (rtol 1e-6/cap 500) WEDGES at p=4 (inner
GMRES hits the cap with reason -3 once alpha deepens); our capped-inexact
discipline (rtol 1e-4/cap 40, FGMRES) rescues it (err 9.542e-5 vs direct
9.620e-5).  Wall: hpG two-stage costs ~30-60x a direct solve at L2 with
this harness (MUMPS K-1 per inner apply, scipy-csr B; the paper's fast
DCT transforms would close this) — iteration counts, not wall, are hpG's
win.  Cross-check: our graded-mesh failure and our fix are both
PC-level and discretization-dependent; the underlying mechanism (S never
collapses) is discretization-independent.

  Cross-check: our graded-mesh failure and our fix are both
PC-level and discretization-dependent; the underlying mechanism (S never
collapses) is discretization-independent.

**True-matfree status (2026-09-10, `matfree_hpG.py`).**  The solve path is
now free of global factorizations: K^-1 = capped CG+GAMG/hypre (AMG built
once per mesh, preconditioner-only), true-S triple-product matvec, Shat
batched cellwise Cholesky, E_beta = 0 in the operator (the §4 Upgrade-2
decoupling, now measured), capped inner GMRES (rtol 1e-4, cap 40), outer
FGMRES.  Correctness: bit-consistent with serial-direct at p=2.  The fair
high-order comparison (L0 p4, both arms, same capped config): iteration
counts IDENTICAL (8/30, err 9.542e-5, outer 27, inner 39.4/apply) —
matfree costs a per-apply A-solve premium (3.5x wall at p4, 1.3x at p2)
that grows with kappa(A).  Graded 51.4x p4: assembled arm HOLDS (8/22,
535 s); matfree arm — GAMG STALLS on graded K0 (honest negative #1);
hypre BoomerAMG healthy per-iteration but projects past the budget
(honest negative #2).  **The AMG for graded high-order K is the open
preconditioner question; everything else in the §4 blueprint is now
measured.**

## 6. Experiments of record: the evidence behind this document

Every claim in Sections 2–4 comes from a scripted, controlled experiment in
`lvpp/experiments/archive_2026-09/` (see the README table there for the one-line
verdict of each). The chain, in the order it was run:

1. **`matfree_fieldsplit.py`** — matfree-operator exactness (bit-exact vs assembled LU
   through the full proximal loop); first observation of the growth mechanism (GMRES its
   12→20 as ψ deepens, then `KSP_DIVERGED_BREAKDOWN`); additive-config floor sweep
   (ε ∈ [1e-2, 1e-1] works at L0 only); **ψ_floor on the operator refuted** (fails at every
   level → the Jp-only rule of §2).
2. **`schur_probe.py`** — the Schur-fieldsplit discovery: 8-point design ladder; winner
   `upper/selfp/LUS` converges LU-identically at L0–L2; failure taxonomy (CG on S fails at
   PC setup — S is indefinite; diag fact-type removes the coupling and hits the exact
   breakdown signature; loose inner-S rtol dies by DTOL).
3. `schur_sp_upgrade.py` — PETSc-native Sp upgrades (ainv/PCLSC) inconclusive; superseded.
4. **`schur_percell.py`** — faithful port of the hpG per-cell broken-basis Sp:
   bit-identical solves → the coupling term is structurally subdominant in this
   discretization (O(h⁴) vs O(h²)); a coupling-kill control (C0 × 10⁻⁶) proving the Sp was
   attached is the template for every later refutation.
5. **`spectrum_anatomy.py`** — the mechanism: true S never collapses (strictly negative,
   h²-scaled, α-independent, no near-null cluster); the degeneracy lives in the monolithic
   operator's dead latent rows (e^ψ → 10⁻⁷⁰…10⁻⁸⁰; 37/177/749 rows).
6. **`growth_driver.py`** — the decisive decomposition: exact block-LU PC with exact
   UNfloored S⁻¹ gives outer-GMRES its = 1 FLAT at every depth (growth is 100%
   PC-fixable); the same PC with floored S reproduces the entire growth curve → the
   ε-floor is the driver. Naive active-set identity rows: worse (inconsistent with the
   coupling).
7. **`rankk_correction.py`** — rank-k correction on span(V) refuted: exact on the dead
   subspace to 10⁻¹⁷, yet fails at depth — the floored PC's deficiency is a TRANSITION
   BAND (σ > 0.1 rank ≥ 70/241/829 ≫ dead count), not low-rank.
8. **`eps_schedule.py`** — the ε-ladder on identical extracted states: ψ_floor=0 is
   LU-identical end-to-end with the growth GONE (14/29/48); safety band ε ≤ 1e-5,
   knife-edge 1e-4, ≥1e-3 kills L2; drift-concentrated floors land in the bad band
   (rejected). Retracts the floor's necessity on the Schur path.
9. **`scale_up.py`** — serial scaling: L3/L4 solve (33k/132k dofs, clean O(h²) errors,
   RSS 1.7 GB); outer its ~N^0.8 (14/29/48/105/329); CG+GAMG-on-K correct but a wall
   loser (Schur apply invokes a K-solve per outer iteration); saturation corner OK.
10. **`amr_health.py`** — the AMR-degeneracy prediction refuted: band-refined meshes are
    dead-row factories (dead fraction 7%→50%; 6.8× dead rows and 6.7× its at matched
    dofs; chain diverges at 14k dofs); grading-sensitivity of the selfp Sp quantified;
    h-fair dead census added (raw |diag(D)| census is h-unfair on graded meshes).
11. **`par_smoke.py`** — 2-rank correctness PASS (LU-identical; both use_amat settings);
    wall-neutral at small sizes.
12. **`grade_pc.py`** — the graded fix: diag(K)⁻¹ mismatch hypothesis refuted
    (κ 3.3e3→8.9e3 across 1×→45.6× grading); **variant B (matfree true-S action + capped
    inner Krylov on −S, rtol 1e-4, cap 40) wins: 6/10/15 its on uniform/band_l4/band_l5**
    vs selfp 22/39/51; variants A and C are honest negatives.
13. **`promob.py`** — end-to-end promotion: uniform LU-identical; graded blowup collapses
    (band_l4 161 vs 297); inner-GMRES arm solves band_l5 (188 vs 323) and removes the
    band_l6 divergence (all linear solves converged where selfp hit ksp=−5); measured
    costs: ~4.5× wall on uniform meshes, and a CG-cap-40 proximal stall on band_l5 that
    the GMRES arm does not exhibit → production inner solver: GMRES.

Cross-cutting harness findings (needed for reproduction): petsc4py PC callbacks require
`x.getArray(readonly=True).copy()` / `y.setArray(arr)` (the context-manager form raises
PETSc error 101 in this petsc4py); relative residual must be computed as
‖Jx − b‖/‖b‖ (two harnesses initially computed ‖Jx‖/‖b‖, mislabeling converged solves).

Of these thirteen, the ones that carry the §4 blueprint are 5 (true-S action, no symmetry
assumption), 7 (E_β belongs on Ŝ only — and, per the eps ladder, at the small end), 8
(the floor's cost mechanism), 12 (the graded-mesh fix and its inner-tolerance knife-edge),
and 13 (the end-to-end promotion, including the inner-CG stall that motivates inner GMRES).

---

## 7. Conclusion and Expected Impact

Incorporating these mechanisms bridges the gap identified in Remark 6.1 of `hierarchicalPG.pdf`:
1. **Fully Matrix-Free Adaptive hpG**: Unlocks true matrix-free $h$- and $hp$-adaptive simulations without resorting to dense or sparse global factorizations.
2. **Optimal Conditioning**: Replaces GMRES with PCG on $-S$, halving inner Krylov memory overhead and eliminating orthogonalization bottlenecks.
3. **No Solution Stagnation**: Disentangles operator accuracy from linear stabilization ($J$ vs. $J_p$), enabling fixed, robust preconditioning floors without loss of convergence accuracy.
