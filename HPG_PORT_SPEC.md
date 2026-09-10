# SPEC: Porting hierarchical Proximal Galerkin (hpG) into the Firedrake LVPP solver

**Purpose.** Implement the solver of Papadopoulos, *hierarchical Proximal Galerkin*
(arXiv:2412.13733v4, "hpG") inside our Firedrake LVPP ecosystem, so that our measured upgrades
can be tested *inside* their framework. Companion docs: `lvpp/RESEARCH.md` (all findings),
`lvpp/HIERARCHICAL_PG_INTEGRATION.md` (bridge + experiments of record), `lvpp/report.html`,
`lvpp/pipeline.html`.

## 0. What the paper specifies (from `ProximalGalerkin/hierarchicalPG.pdf`, v4 2026-08-06)

### 0.1 Spaces (§3)
- \(u_h \in U_{h,p}\): H1-conforming hierarchical basis = P1 hat functions + per-cell Jacobi
  bubble functions \(W_n = (1-x^2)P_n(x)/(2(n+1))\), partial degree \(p\), tensor-product in 2D.
  **As a polynomial space this is exactly standard continuous \(P_p\) on the mesh** — the hierarchy
  is a *basis* choice (bubble-bubble stiffness diagonal in 1D; dW_n/dx = −P_{n+1}).
- latent \(\psi_h \in \Psi_{h,p}\): **L2-conforming discontinuous** per-cell translated Legendre
  polynomials, degree \(p-2\) for the obstacle problem (inf-sup pair [60, Lem B.3]);
  degree \(p-1\) for gradient-type. **Consequence (Lemma 3.2): the mass matrix of Ψ is
  diagonal**, and the Newton block \(D_\psi\) is (after permutation) **block-diagonal per cell**
  with dense \(O(p^d)\) blocks, invertible cellwise.
- In 2D/3D the spaces are tensor products of the 1D ones → the mesh must be **quadrilateral**
  (tensor-product cells). This is a stated limitation of their framework (non-Cartesian cells
  not supported).

### 0.2 Newton system (§4.1, eq 4.1-4.2)
$$G = \begin{bmatrix} A_\alpha & B \\ B^{\top} & -D_\psi - E_\beta \end{bmatrix},\quad
A_\alpha = \alpha A,\quad [D_\psi]_{ij} = (\zeta_i, e^{-\psi_h}\z_j)_{L^2},\quad
B_{ij} = (v_i, \z_j)_{L^2},\quad [E_\beta]_{ij} = \beta(\z_i,\z_j)$$
- Sign convention: their \(D_\psi\) is a POSITIVE mass ⊙ e^{−ψ} (their ψ is the negative of ours).
  \(S = -(D_\psi + E_\beta + B^{\top}A_\alpha^{-1}B)\) is then negative definite (same as ours).
- **E_β is formulation-level stabilization, in the operator G** — for the obstacle problem
  "in most examples in Section 6, β = 0" (E_β ≡ 0); it is essential only for gradient-type
  constraints (§4.2, where β = 10⁻⁵ was used because S is nearly singular otherwise).
- \(A = \) stiffness is **α-independent** → ONE cached Cholesky serves the whole proximal loop
  via \(A_\alpha^{-1} = \alpha^{-1}A^{-1}\). This is a structural advantage our lvpp config shares
  (measured: K = α·stiffness).
- \(D_\psi\) is the only iterate-dependent block.

### 0.3 Solver (§4.4-4.5, Fig 3)
- Outer **FGMRES** on the monolithic G, preconditioner choices P_F (full triple-product
  factorization; sequential form eq 4.5: two A⁻¹ applies + one S⁻¹), P_{LD}, P_D (one A⁻¹ + one S⁻¹).
- **S⁻¹ via inner GMRES** preconditioned by Ŝ = −D_ψ − E_β − B̂ᵀÂ_α⁻¹B̂ where Φ_{h,p} is the
  per-cell **Legendre spectral-Galerkin basis** \(Y_n = P_n - P_{n+2}\) (vanishes at cell endpoints):
  Â_α = α(∇_h η_i, ∇_h η_j) → block-diagonal (4N blocks in 2D), B̂ = (η_i, ζ_j) mass between Φ and Ψ.
  Ŝ is block-diagonal with dense per-cell blocks → cellwise factorization.
- Fig 4 (obstacle, worst case D_ψ ≡ 0, β = 0): Ŝ-preconditioned GMRES on S shows
  **polylogarithmic growth in p and 1/h**. Cache Cholesky of A once per nonlinear solve
  (α-independent). In 2D they always use the cached Cholesky + P_F.
- §5 fast-implementation notes: fast Legendre↔Chebyshev/DCT transforms; three remedies for
  ill-conditioning at large p/h (block PC, E_β small, small α allowed — "the solver converges
  without requiring α → ∞"); non-standard quadrature for D_ψ (expansion in the Ψ basis,
  breaking symmetry); inexact Newton linear solves (theory open).

### 0.4 What is NOT matrix-free about hpG
A, B, E_β are **assembled** (sparse, O(p^d/h^d) nnz); the Cholesky of A is computed and cached;
Ŝ is assembled and block-factorized. Only the true-S action (triple product) is matrix-free.
"Matrix-free" in the libCEED sense (pointwise kernels, no assembly) is NOT their claim — their
claim is sparsity promotion: high-order basis → the operator's density is localized in the
nontrivial block. Port faithfully to the paper first (assembled A/B/D, cached Cholesky),
then test the matfree upgrades on top.

## 1. What we bring (measured, from RESEARCH.md Finding 6 + companion blocks)
1. Floorless Schur configuration on equal-order CG×CG: LU-identical solves L0-L4 (545→132k
   dofs), max outer its 14/29/48/105/329 (~N^0.8).
2. The ε-floor is unnecessary on the Schur path and was the whole growth driver; ε-ladder:
   ≤1e-5 safe, 1e-4 knife-edge, ≥1e-3 kills L2.
3. Graded/AMR meshes: selfp blows up (297/323 its, ksp=-5 divergence at 14k dofs); the fix
   is the matfree true-S action S v = Dv − Bᵀ(K⁻¹(Bv)) with capped inner Krylov on −S
   (rtol 1e-4, cap 40): 6/10/15 cold on uni/band_l4/band_l5; end-to-end 161/188, band_l6
   divergence eliminated.
4. Inner-CG stall at band_l5 (proximal increment plateau, all linear solves 4-5 its) —
   inner GMRES does not stall. So the inner solver choice matters nonlinearly.
5. petsc4py harness rules: PC callbacks use `x.getArray(readonly=True).copy()` /
   `y.setArray(arr)`; rel-resid = ‖Jx−b‖/‖b‖; options_prefix alnum/underscore only.
6. AMR machinery available: viamr SBR band-refined meshes (amr_health.py archive pattern)
   — the graded-mesh generator for the robustness test.

## 2. What to implement

### Stage 0 — faithful hpG discretization in Firedrake (gate for everything)
- **Mesh**: quadrilateral tensor-product mesh. `RectangleMesh(16, 16, 2.0, 2.0, ..., quadrilateral=True)`
  (the crossed-triangle benchmark is replaced; document the change — hpG is tensor-product).
- **Spaces**: `V_u = FunctionSpace(mesh, "CG", p)` (u; the hierarchical basis spans the same
  polynomial space — state this explicitly in the code comments); latent `V_psi = FunctionSpace(mesh, "DG", p-2)`
  (obstacle pairing [60, Lem B.3]). Check whether this Firedrake supports a Legendre-modal DG
  variant (`FunctionSpace(mesh, "DQ", k, variant=...)` — probe FIAT); if yes use it (diagonal
  ψ-mass per Lemma 3.2 becomes exact), else use the default nodal DG and document the
  difference (block-diagonal D_ψ per cell survives in any DG basis; the diagonal-mass
  elegance is lost).
- **lvpp integration**: instantiate `LVPP` with `psi_spaces=[V_psi]` (the kwarg exists for
  non-equal-order latent spaces), bounds=(ψ_obs, None), energy = 0.5*|∇u|² dx (f=0 sphere),
  α_max 10 double-exponential, tol 1e-4, l2 linesearch, snes_rtol 1e-6, quadrature_degree
  raised (≥ 2p). **RISK**: lvpp internals may assume the latent space equals a CG copy of the
  primal space (u_tilde evaluation, drift field spaces, AMR census) — audit `lvpp.py` for
  these assumptions, document every one that breaks, and work around WITHOUT editing lvpp.py
  (e.g. project e^φ into CG for observable outputs). If a fundamental incompatibility is
  found (e.g. psi_spaces not honored), that is a legitimate stage-0 stop with a written diagnosis.
- **Verification (stage-0 gate)**: (a) ψ-mass matrix diagonal (Legendre variant) / block-diagonal
  per cell (any DG basis) — assemble and check; (b) D_ψ = (ζ_i, e^{-ψ_h}ζ_j) block-diagonal per
  cell — assemble and verify off-cell-block entries ~0; (c) serial direct MUMPS solve of the
  monolithic G on L0/L1 quads converges to the analytic sphere solution (psiUFL/uexactUFL —
  errors are comparable ACROSS discretizations because the analytic solution is shared);
  record err(u_h) vs our P1 references (they will differ — hpG at p≥2 should be MORE accurate
  per dof; that is the paper's selling point, measure it).

### Stage 1 — hpG solver machinery
- Cache the Cholesky (PETSc preonly+lu or scipy splu) of the α-independent stiffness A on CG_p;
  A_α⁻¹ v = (1/α) A⁻¹ v. (If firedrake/CG_p on quads makes A = αA + BC-rows only, verify no
  other α-dependence sneaks in — compare assembled K at two different α values, elementwise.)
- Build Ŝ per §4.5: Φ = per-cell spectral-Galerkin basis (Y_n = P_n − P_{n+2}; in Firedrake:
  probe for a Legendre-modal DQ variant; fallback = default DG_p basis with the documentation
  note that Y_n's endpoint-vanishing property is what makes Â block-diagonal — a plain DG_p
  basis does NOT make Â block-diagonal! If the Legendre variant is unavailable, the fallback
  is to build Ŝ via explicit per-cell assembly loops using Firedrake cellwise TensorProductElement
  of 1D "DQ Legendre" if supported, or via scipy: assemble B̂ rows by evaluating the Φ basis
  per cell. Document the chosen route and its cost.); factor Ŝ cellwise (dense per-cell LU via
  batched scipy/petcs or block-diagonal sparse LU).
- Assemble B̂ = (η_i, ζ_j) between Φ and Ψ (both DG → per-cell blocks).

### Stage 2 — the two-stage solver (paper eq 4.5)
- Outer FGMRES on monolithic G (assembled G or matfree G action: A_α⁻¹ via cached Cholesky +
  scaling; D_ψ matvec per-cell; E_β = 0).
- P_F apply: y = b_ψ − Bᵀ A_α⁻¹ b_u; δψ = S⁻¹y via **inner GMRES preconditioned by Ŝ**
  (block-diagonal factorization; rtol_inner and max_it as paper: converge S-solves to ~1e-6
  for near-exact outer convergence); δu = A_α⁻¹(b_u − B δψ). Prefer P_F in 2D (paper §4.4).
- Measure on uniform quad meshes L0-L2 (16², 32², 64² cells; p = 2 and p = 4):
  prox/newton counts, err(u_h) vs analytic, outer-FGMRES its, inner-GMRES its per apply,
  wall. Compare against: our P1 numbers (8/21, max its 14/29/48) and the paper's published
  14-36 avg its (their Table 1-2 setting, uniform Cartesian, obstacle).

### Stage 3 — OUR upgrades, tested INSIDE hpG (the actual point)
1. **J/Jp decoupling**: E_β = 0 in the operator always (obstacle case already β=0 in the
   paper; keep it that way and verify — our Finding-1 rule says operator stabilization is
   forbidden). For Ŝ-only stabilization, sweep β_Ŝ ∈ {0, 1e-5, 1e-4, 1e-3} on the worst-case
   state (D_ψ ≡ 0 analogue: first proximal solve where ψ is shallow — and at converged depth)
   and map the paper's Fig-4 claim onto our meshes.
2. **Graded meshes**: rebuild the amr_health band chain (viamr SBR) on the quad mesh;
   run (i) hpG two-stage solver as published (their untested-on-graded claim), (ii) hpG with
   our inexact-capping discipline (inner GMRES rtol 1e-4, cap 40, outer FGMRES), and compare
   against our measured selfp blowup (297/323/diverged) and our variant-B fix (161/188/l6-ok).
   This is the experiment that decides whether our upgrades "measure up" — their polylog
   claim on graded meshes is untested (Remark 6.1).
3. **Matfree A-action arm** (stretch): replace the cached Cholesky by CG+GAMG on A_α (α-scaled)
   and re-measure — the paper's §6.5 analogue in our ecosystem.

### Stage 4 — writeup
Verdict paragraph + table answering: (a) does faithful hpG-in-Firedrake reproduce the paper's
iteration behavior (polylog, 14-36 its)? (b) does our floorless finding transfer (β_Ŝ = 0 vs
the paper's E_β philosophy)? (c) does hpG's Shat survive graded meshes (their untested claim)
or does our measured selfp-style blowup recur — and does our capped-inner-Krylov discipline fix
it inside hpG too? (d) accuracy per dof: hpG (p≥2) vs our P1 at matched dofs.

## 3. Constraints (hard)
- Env guard on EVERY python invocation: `PETSC_DIR=/home/stefano/firedrake/petsc
  PETSC_ARCH=arch-firedrake-default OMP_NUM_THREADS=1` with
  `/home/stefano/firedrake/venv-firedrake/bin/python`.
- Do NOT modify any existing file: lvpp.py, RESEARCH.md, REPORT/Pipeline HTML docs,
  HIERARCHICAL_PG_INTEGRATION.md, experiments/** (the archive is frozen), tests/, viamr/,
  examples/. New files live in `lvpp/experiments/hpg/` (this is a fresh folder; the archive
  is frozen).
- petsc4py PC/KSP callback idiom: `x.getArray(readonly=True).copy()` / `y.setArray(arr)` —
  context-manager form raises PETSc error 101 in this petsc4py.
- PETSc options labels as options_prefix: alphanumeric/underscore only. No petsc4py.init().
- Serial only (the MPI datum is already recorded). No pytest/formatters.
- Wall budget: 75 min across all stages. Stages are gated: stage 0 verification is mandatory
  before stage 1; a failed gate = stop and report the diagnosis (that is a valid deliverable).
- The lvpp.py audit (ψ ∈ DG support) may reveal hard blockers: if lvpp cannot run with a DG
  latent space without edits, implement the hpG discretization as a STANDALONE solver script
  (assemble G directly, own Newton loop on the two-row residual with firedrake's
  NonlinearVariationalSolver on the mixed space) rather than editing lvpp.py — document the
  divergence from the lvpp class. The comparison to our measured numbers stays valid because
  both use the same benchmark/analytic solution.
- Honest negatives are valid deliverables; never fabricate numbers. Bit-identical /
  controlled comparisons are the gold standard.

## 4. Acceptance
- Files under `lvpp/experiments/hpg/` (harness + solver), running under the env guard.
- Stage-0 verification table: ψ-mass diagonal/block-diagonal; D_ψ cell-block structure;
  direct-solve control vs analytic sphere solution (err(u_h) per dofs, p=2 vs p=3).
- Stage-2 table: per level (L0-L2 quad), per solver arm: prox/newton, err(u_h), outer-FGMRES
  its, inner-GMRES its/apply, wall — vs our P1 floorless numbers and the paper's published range.
- Stage-3 tables: β_Ŝ sweep (worst-case + converged states); graded band chain on hpG:
  selfp... i.e. hpG-published discipline vs our capped-inner discipline, per level, with the
  band_l6 divergence check.
- Final verdict paragraph: (a) faithful-port status, (b) does our upgrade set measure up
  INSIDE hpG (floorless/beta-on-Shat-only/capped-inexact), (c) what hpG's framework gives us
  that our config lacks (per-dof accuracy at p≥2, polylog robustness claims) and what ours
  gives that theirs lacks (measured degeneracy mechanism, AMR tradeoff, parallel datum,
  Jp-only rule).