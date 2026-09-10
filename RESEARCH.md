# LVPP research notes: matrix-free structure, contact degeneracy, a posteriori estimation

Working notes, 2026-09-03. Everything here is measured against the code in
`firedrake/lvpp` (the LVPP solver) and `firedrake/viamr` (AMR + estimators),
with the sphere obstacle problem as the common benchmark. Line numbers are
approximate as of this date; symbols are stable.

References (papers):
- **[LVPP]** Dokken, Farrell, Keith, Papadopoulos, Surowiec,
  *The latent variable proximal point algorithm for variational problems with
  inequality constraints*, arXiv:2503.05672 (2025). Source of the algorithm
  (eq. 2.7), the alpha rules (eq. 3.8), the Legendre-family catalog, and the
  dolfinx reference implementation.
- **[KG24]** Keith, Surowiec, *Proximal Galerkin: a structure-preserving finite
  element method for pointwise bound constraints*, Found. Comput. Math. (2024).
  The entropy-duality predecessor; source of the cap obstacle and the
  `phi`-on-quadrature-element trick in the reference obstacle example.
- **[NSV03]** Nochetto, Siebert, Veeser, *Pointwise a posteriori error control
  for elliptic obstacle problems*, Numer. Math. 95:163-195 (2003).
- **[NSV05]** Nochetto, Siebert, Veeser, *Fully localized a posteriori error
  estimators and barrier sets for contact problems*, SINUM 42(5):2118-2135
  (2005).
- **[HIK02]** Hintermüller, Ito, Kunisch, *The primal-dual active set strategy
  as a semismooth Newton method* (2003, SIAM J. Optim. 13(2)) — what
  `snesvinewtonrsls` is (active-set/semismooth; row mutation per iteration).
- **[Caf77/Caf98]** Caffarelli, free-boundary regularity for the obstacle
  problem; u ∈ C^{1,1}, gap ≍ dist²  from a smooth obstacle (Caffarelli 1977;
  "The obstacle problem revisited" 1998).
- **[CEED]** the libCEED project and its BP benchmarks (Brown et al.) —
  fixed pointwise kernels, no combinatorial operator changes; the performance
  model motivating the matrix-free question.

---

## Code map

**Note (2026-09-10):** all session experiment scripts moved to
`experiments/archive_2026-09/` (with an experiment-of-record README table
inside); script paths in this document read `experiments/<name>.py` →
`experiments/archive_2026-09/<name>.py`.

| piece | location |
|---|---|
| LVPP solver class | `lvpp/lvpp/lvpp.py`, class `LVPP` (~line 313) |
| proximal loop, warm start, drift capture | `LVPP.solve` (~line 489; `warm_start` kwarg; drift capture inside the accepted-iterate block) |
| Jp (preconditioner-Jacobian) construction | `LVPP.__init__`, `jacobian_regularization` hook (`Jp = J + reg`) |
| proximal drift = discrete multiplier | `LVPP.drift` (per-accepted-iterate `(psi_prev - psi)/alpha`), captured before the psi_prev shift |
| Legendre families | `lvpp/lvpp/legendre.py` (ShannonLower/Upper, FermiDirac, Hellinger, GibbsSimplex) |
| Constraint interface | `lvpp/lvpp/constraints.py` (`Constraint`, `BoxConstraint`; `coupling_form`/`state_form`/`observable` = the family-generic surface) |
| NSV03 with dual injection | `viamr/viamr/viamr.py`, `nsv03mark(..., dual=None)` (~line 1045; injection branch ~1153; per-term debug print behind `VIAMR(debug=True)`) |
| NSV05 (no dual injection; see F3) | `viamr/viamr/viamr.py`, `nsv05mark` (~1375) |
| method sweep + saddle estimator | `viamr/examples/sphere_lvpp.py` (`saddle_estimator`, `-methods uni,udobr,nsv03,saddle`) |
| adaptive single-run draft | superseded by the sweep; kept in git history of `viamr/examples/sphere_lvpp.py` |
| uniform study (reference config) | `lvpp/examples/sphere_lvpp.py` |
| matrix-free experiment | `lvpp/experiments/matfree_fieldsplit.py` |

Solver configuration that works everywhere (uniform study, sweeps):
`snes_rtol 1e-6`, l2 line search (`maxlambda 1.0`), `alpha_max 10`,
H1-increment exit tol 1e-4, quadrature degree 6.  The `newton_adaptive`
alpha rule is `lvpp`'s default-named alias for the fracture schedule in
[LVPP].

---

## Finding 1 (validated): matrix-free exactness + degeneracy-aware preconditioning

**Claim.** The LVPP saddle system is structurally ideal for matrix-free
(libCEED-style) operators: fixed sparsity, pointwise-only nonlinearities
(`exp(psi)`, `tanh(psi/2)`), no active-set row mutation, ever.  What the
active-set approach (`snesvinewtonrsls`, [HIK02]) does by dropping/re-adding
rows each Newton iteration, LVPP does by a smooth change of variables
(`u = phi + exp(psi)` is feasible for *every* psi), so the operator never
changes.  The price: a degenerate diagonal, see below.

**Evidence.** `lvpp/experiments/matfree_fieldsplit.py`:
- Amat = matfree, Pmat = assembled, KSP = preonly + LU on the Pmat:
  reproduces the assembled-LU baseline **iteratively identically** —
  level 0: 8 prox / 21 Newton, err 1.553e-2 (baseline identical);
  level 1: 11 prox / 23 Newton, err 3.599e-3 (identical).
  The matfree action is exact; the proximal loop, warm start, and alpha
  schedule run through without operator change.
- Amat = matfree + `PCFieldSplit` (additive: u-block CG+GAMG, psi-block
  Jacobi) with an eps floor on the psi diagonal converges at moderate scale
  (level 0: 7 prox / 22 Newton, err 1.553e-2) and fails beyond.

**Mechanism of the degeneracy** (the important part; see "free-boundary
relation" below for the refinement question):
- psi-row: `u - phi - exp(psi) = 0` forces `exp(psi) = u - phi`.  On the
  contact interior (extended region where u = phi), the gap is 0 to machine
  precision after enough proximal iterations, so exp(psi) underflows and the
  (2,2) Jacobian block D = diag(exp(psi)) has **exactly zero rows** there.
- The psi *depth* is mesh-independent: psi runs away linearly at rate
  alpha·lambda per accepted proximal iteration (lambda ≈ -Delta(phi) - f on
  the cap ≈ 4 at the center), and proximal counts are mesh-independent
  (8-15).  Measured depth: -150 to -430 at level 0 (alpha_max 10).
- The count of exactly-degenerate nodes grows ∝ contact area ∝ h^-2.
- Across the free-boundary transition band (width ~ h), the D diagonal spans
  its whole dynamic range (gap from ~0 to O(1) within one element layer;
  gap ~ dist^2 by [Caf98]-type regularity for a smooth cap, so the smallest
  positive band value scales like h^2).  The band is intrinsically the
  worst-conditioned feature at every resolution; refining it makes it
  thinner and steeper, not better.

**Preconditioner consequence (measured).**  GMRES iteration counts on the
matfree system grow 12 → 14 → 17 → 18 → 20 as psi drifts, then
DIVERGED_BREAKDOWN, on systems that LU solves fine.  With the additive
block-diagonal PC the coupling block M K^{-1} M is ignored; where
D → 0 the preconditioned operator crosses the origin (the Schur complement
is D - (1/alpha) M K^{-1} M ≈ -(1/alpha) M K^{-1} M < 0 while the PC assumes
a D-dominated diagonal).  The needed floor therefore scales with the
coupling strength ~ 1/alpha_k and with the local multiplier — hence the
experiment below.

**Design rule discovered en route (now in `lvpp`).**  The eps floor must
live on **Jp** (the preconditioner Jacobian), never on J: putting it on J
(a) perturbs the Newton direction (DIVERGED_DTOL with exact LU PC at any
eps > 0), and (b) breaks the matfree action path itself — added 2-forms
with mixed-indexed arguments *and* whole-mixed-space forms both
mis-evaluate under `mat_type: matfree` while assembling correctly.
`LVPP.__init__` now passes `Jp = J + reg` to
`NonlinearVariationalProblem(..., Jp=Jp)`; the Newton residual and Jacobian
stay exact.  (Bug-report candidate for firedrake: matfree action of added
2-forms on mixed spaces.)

**Relation to free-boundary-aware meshes.**
- The exactly-degenerate set is the contact *interior*, not the free
  boundary.  Free-boundary-aware marking (viamr principle: keep the
  known-active interior coarse, refine the boundary ring + inactive set)
  therefore caps the number of machine-zero D-rows at O(perimeter/h)
  instead of O(area/h²).  Uniform refinement is the degeneracy-*maximizing*
  strategy at matched dof budget; boundary-oriented AMR is the
  degeneracy-*minimizing* one.
- The coarse-mesh pathology (Newton parking psi at wrong-branch compromise
  values near a smeared free boundary; measured level-0 err(u_tilde) = 114)
  is the *opposite*: it improves with resolution because a thin band pins
  each node to contact-or-detached cleanly.
- Net: free-boundary-aware AMR helps accuracy (your principle) and helps
  the preconditioner (fewer dead rows, no wasted interior refinement).
  Uniform refinement is simultaneously the worst case for both.

**Measured (2026-09-04, `experiments/matfree_fieldsplit.py`, sphere 3
levels, LU reference errors 1.553e-02 / 3.599e-03 / 8.375e-04).**  The
floor sweep settles the "Open" question with a mixed verdict:

| configuration | L0 (545) | L1 (2113) | L2 (8321) |
|---|---|---|---|
| LU, Jp = J (reference) | 8/21 ✓ | 11/23 ✓ | 8/19 ✓ |
| matfree + Jp floor 1e-3..1e-1, ksp_rtol 1e-6 | ✓ 1e-2..1e-1 (7/22, LU-exact err); 1e-3 too small | ✗ | ✗ |
| drift floor eps0 + c·alpha·|lambda|, eps0 = 1e-3 | ✓ c = 0.03 (7/22); c ≥ 0.1 ✗ (overshoots early prox steps where alpha·|lambda| is large) | ✗ | ✗ |
| same, ksp_rtol 1e-2 (inexact Newton) | ✓ 8/45 | ✗ (knife-edge) | **✓ 8/65, err 8.376e-04 (LU-exact)** |
| Eisenstat-Walker (v2, ±use_step) | ✓ 7/22 | ✗ | ✗ |
| floor on the *operator* (psi_floor_operator) | ✗ all levels, all tolerances | ✗ | ✗ |
| schur fieldsplit (any) | PC_FAILED at setup | ✗ | ✗ |

**Verdict.**  (1) The Jp-only floor is *necessary and correct*: putting it
on the operator destroys the proximal path at every level (direct
evidence for the design rule above — the root F = 0 is unaffected but the
Newton path is not; the proximal dynamics need exact Newton steps).  (2)
At level 0 the constant-floor window is wide (1e-2..1e-1) and the
drift-proportional floor works at its calibrated scale c ≈ 0.03 with a
10× smaller base.  (3) At depth, no *preconditioner-only* floor survives
tight inner solves.
The mechanism is now measured (spectral anatomy below): the operator has
a near-null cluster of deep-latent modes — eigenvalues 6.4e-7 / 4.6e-8 /
3.2e-9 across levels, eigenvectors >90% latent mass localized where
phi < -50, mode count proportional to the deep-contact node count
(36 / 177 / 760 = O(area/h^2)) — so GMRES's attainable relative residual
floor is ~sqrt(kappa)·eps (~1e-4 at L0, ~1e-3..1e-2 at L2),
restart-independent.  Tight rtol 1e-6 is below that floor at depth:
KSP_DIVERGED_BREAKDOWN is floating-point stagnation, not a preconditioner
defect.
Fixed loose inner tolerance (rtol 1e-2) rescues level 2 with LU-exact
errors; level 1 is a documented knife-edge (fails at every setting
including EW — the Newton path chaos near the degeneracy boundary, not a
floor-magnitude problem).  (4) Practical recipe for the libCEED port:
matfree Amat + assembled floored Pmat + fieldsplit (additive,
`pc_fieldsplit_use_amat: False` is mandatory) + inexact-Newton inner
rtol ~1e-2; expect Newton its to roughly double (22 → 65 at L2) — the
per-iteration cost is QFunction-cheap, so the trade is favorable.
**Scaling hypothesis (falsified 2026-09-04).**  The candidate fix — a
change of variables delta_phi = e^{-phi}·w realizing the symmetric
congruence T J T (T = diag(e^{-phi/2})) as right preconditioning
M^{-1} = diag(e^{-phi}) — is disproved by dense spectral extraction of
the deep-state Jacobian: the latent *columns* of J are dominated by the
e^phi-free drift coupling d(lambda)/d(phi) = -1/alpha, not by
e^phi-barrier terms, so the scaling amplifies them by up to 1e157
(eigenvalues of J·diag(e^{-phi}) reach 2.4e157) while unscaled J at the
converged state has kappa only ~1.2e8 (L0) / 1.8e9 (L1) / ~2.5e10 (L2,
from min|eig| 3.2e-9 via ARPACK).  Column equilibration
(right-PC Jacobi, diag(J)^{-1}) fails for the same reason, and a pure
diagonal PC on either side cannot replace the block fieldsplit: it
stagnates (~0.5%/iteration) already at the benign first proximal solve.
The productive residue of the falsification is the mechanism above: the
attainable-GMRES-accuracy floor sqrt(kappa)·eps, and the target for any
future cure — deflate/project the analytically-known deep-latent cluster
(latent basis functions on phi < -50 nodes) so tight rtol becomes
attainable; that is the top follow-up experiment.

**Resolution update (2026-09-08).**  The "Schur approximation that
captures M K^{-1} M cheaply" question is now ANSWERED in practice — see
**Finding 6**: schur/upper/selfp fieldsplit with LU on the assembled
approx-Schur Sp converges at every depth with LU-identical Newton
counts; the spectral anatomy shows the true S is healthy (h²-scaled,
no near-null cluster), so the remaining outer-GMRES growth is purely
Sp-approximation quality.  Still open: better Sp (block_diag/full ainv,
PCLSC), deflation of the dead subspace (top priority for tight rtol at
depth), MINRES symmetric reformulation, level-1 knife-edge diagnosis.

---

## Finding 2 (validated): psi-based contact marking through u_tilde

`u_tilde - lb = exp(psi)` **exactly nodally** (both are CG1 interpolants of
expressions sharing the psi field), so viamr's standard
`elemactive(u_tilde, lb)` marks `{exp(psi) < activetol}` — a dual-variable
contact test through the primal-facing API.  For LVPP solutions the naive
nodal comparison on u_h is dead (u_h is an L2 projection of the
reconstruction; it misses the bound pointwise by O(h²), measured feas(u_h)
= 1.3e-3 on the sphere while feas(u_tilde) ≡ 0).  Measured J(A_exact,
A_psi) = 0.33 → 0.98 across the adaptive sweep.

**Framing note (agreed):** the marking pipeline itself is unchanged viamr
machinery; LVPP only changes what feeds it.  The active-interior error is
obstacle interpolation error (a feature; no refinement needed there), so
the honest scoreboard is contact identification (J) and inactive-set error,
not global L2.

---

## Finding 3 (validated): the saddle-point residual estimator

Built purely from `coupling_form`/`state_form` residuals — therefore
family-generic by construction, unlike NSV03/NSV05:

    u-row:   -div(grad u) - f - lambda = 0,  lambda = (psi_prev - psi)/alpha
    psi-row:  u - phi - exp(psi)             = 0

eta_K² = hK²(R_u² + [∂n u]²) + R_psi²  (R_psi unscaled: mass-type row).
Implementation: `saddle_estimator` in `viamr/examples/sphere_lvpp.py`
(lambda from `LVPP.drift`; facet jumps assembled cell-diagonally following
viamr's `gradrecinactivemark` pattern).

Measured: best-per-dof adaptive method (err 3.99e-4 @ 14k dofs vs udobr
1.37e-3 @ 23k); the marked-in-contact fraction quantifies the
"do-not-refine-known-active" principle: 100% of marks inside the exact
contact at level 0 (the lambda-driven u-row residual is O(1) there and is
absorbed by the constraint — not error), rebalancing to ~33%.

Caveats: no reliability theory (empirically validated indicator only);
the lambda-on-contact-isn't-error weighting question is open (per Finding
1's principle, the u-row lambda term should arguably be excluded from
refinement inside the known-active region).

---

## Finding 4 (validated): two structural incompatibilities with legacy NSV

1. `nsv05mark` asserts its input is *simultaneously* pointwise-admissible
   (`u_h - lb >= 0` elementwise) and a discrete-VI solution (nodal residual
   functional s_z <= 0 off contact).  LVPP splits these roles: u_tilde is
   admissible but not a VI solution (positive flux-jump functional), u_h
   carries the VI structure but misses the bound pointwise.  Each assertion
   passes on a different function; `nsv05mark` cannot run unmodified on
   either.  (viamr.py lines ~1616 and ~1649 in the pre-edit file.)
2. `nsv03mark` with the exact dual injected as sigma_h: the drift is
   strictly positive over the whole contact, so term 3's star-dilated
   detachment region omega_bot swallows the contact and the term counts the
   continuum-obstacle *interpolation deficit* there (per-term breakdown,
   `VIAMR(debug=True)`: blockgaplo 0.13 → 1.29 dominates Eh; resid ~ 1e-2).
   Effectivity Eh/err_Linf grows 1.2 → 38 and err(u_h) stalls at 7.3e-3 —
   refinement consumed exactly where u = psi is known.  The drift cannot
   replace sigma_h in term 1's residual either (X = |f + tactive·sigma_h|
   would be O(1) on contact — conflation of multiplier with residual).
   Making NSV03 dual-aware (restricting omega_bot, or the Finding-2-style
   residual folding) is open; the saddle estimator is the working
   LVPP-native alternative.

---

## Finding 5 (validated): robustness envelope

- The saturation corner: psi drifts to -inf at rate alpha·lambda·k on
  exactly-active extended regions; past |psi| ~ 350-700 the entropy
  Jacobian rows underflow (sech²/exp), Newton rows become exactly zero
  (MUMPS PC_FAILED observed at |psi| = 1351).  Documented in the
  `LVPP` docstring Note; mitigations: moderate tol, alpha_max, l2 line
  search, Jp floor.
- Warm starts: `solve(warm_start=True)` amortizes re-solves across AMR
  levels (4-15 proximal iterations measured; without warm start the level-5
  u_tilde error was 6.9e+08, with it 3.3e-2).
- The paper's disk-domain/homogeneous-BC variant does not hit the corner as
  hard as the square-domain/uexact-BC variant used here; the envelope is
  data-dependent (lambda magnitude vs alpha schedule).

## Finding 6 (validated): Schur fieldsplit solves at depth — the scalable
matfree configuration exists

**Configuration (winner of an 8-point design ladder,
`experiments/schur_probe.py`, independently reproduced by the parent).**
matfree Amat + assembled floored Pmat + `pc_fieldsplit_type: schur`,
`pc_fieldsplit_schur_fact_type: upper`, `pc_fieldsplit_use_amat: False`,
`fieldsplit_0`: preonly + MUMPS-LU (exact K),
`pc_fieldsplit_schur_precondition: selfp`, `fieldsplit_1`: preonly + LU of
the assembled approximate Schur
Sp = D - B diag(K)^{-1} B^T (O(nnz) to build), `psi_floor: 1e-2`.
Outer KSP: GMRES rtol 1e-6, restart 250.

**Result: Newton/proximal counts and errors IDENTICAL to the LU reference
at every level** (parent rerun, 2026-09-08):

| level | dofs | prox | newton | err(u_h) | wall | LU ref |
|---|---|---|---|---|---|---|
| 0 | 545 | 8 | 21 | 1.553e-02 | 1.1 s | 8/21/1.553e-02 |
| 1 | 2113 | 11 | 23 | 3.599e-03 | 11.9 s | 11/23/3.599e-03 |
| 2 | 8321 | 8 | 19 | 8.375e-04 | 81.1 s | 8/19/8.375e-04 |

**Remaining scalability limiter (measured).**  Outer GMRES its per Newton
solve grow ~45 (L0) -> 130-560 (L1) -> 400-926 (L2, near max_it 1000):
Sp is only a *diagonal*-approximation of the true Schur S (diag(K)^{-1}
vs K^{-1}), and its quality degrades with h.  Next lever (PC-config only,
no lvpp.py changes): `mat_schur_complement_ainv_type block_diag|full`, or
PCLSC for Sp; then replace MUMPS-LU on K with CG+GAMG (already validated
on the healthy block).

**Design-ladder evidence (level 0).**  `diag` fact-type fails with the
exact breakdown signature (outer GMRES ksp=-5 -> snes=-3: coupling
removed again); `lower` dies via DIVERGED_DTOL step blowup; loose inner
S-solves (rtol 1e-2, 1e-4) die via DTOL; inner rtol 1e-6 converges;
exact LU on Sp is strictly better and level-robust.  GMRES (never CG) on
block 1 is necessary and sufficient for PC setup (S is indefinite).
`user` == `a11` exactly (PETSc falls back to A11 without a user matrix;
verified in PETSc sources).

**Spectral anatomy** (`experiments/spectrum_anatomy.py`,
`experiments/spectrum_results.md`; assembled monolithic J at early /
3-proximal / converged states, levels 0-2; true Schur S = D - B^T K^{-1}B
applied matrix-free via LU on K; ARPACK + 120-step Lanczos):

1. Block structure verified, not assumed: for `bounds=(psi, None)`
   (LOWER bound) D = -M⊙exp(psi): a mass matrix scaled pointwise — NOT
   diagonal (off-diag O(median diag)); B = M; K = alpha·stiffness
   (diag median 4·alpha — K scales WITH alpha).  Any Schur-PC design
   must treat D as a mass-matrix block, not a Jacobi block.
2. Degeneracy drift (paper-figure data): phi_min -3.5 (early) -> ~-30
   (3 prox its) -> -318..-763 (converged); |D|-diag min collapses
   1e-4 -> 1e-11 -> 1e-28..1e-80; dead rows n(|d|<1e-12) at convergence
   = 37 / 177 / 749 across levels 0/1/2 — scales with contact area
   (area/h²), the AMR link.
3. **The true Schur S never collapses**: strictly negative definite,
   lambda_S_min ~ -5.6e-2 / -1.4e-2 / -3.5e-3 (pure h² scaling,
   alpha-independent), essentially invariant from shallow to fully
   degenerate states; near-zero count in S is 0/1/1 — NO near-null
   cluster in the true Schur at any depth.  The GMRES breakdown is
   therefore NOT a Schur-rank problem: it lives in the monolithic
   operator's degenerate latent rows (and in any PC whose D-block is
   asked to invert them unfloored).  This CORRECTS the earlier
   "two-block near-dependence / S collapsing" reading in the
   equilibration-refutation note below: the near-null space is the
   D-rows' flat manifold; S itself is healthy and h²-scaled.
4. Floor mechanism measured: the lvpp psi_floor form assembles to a FULL
   mass matrix (off-diag 2.6e-5 / 6.5e-6 / 1.6e-6 at L0/L1/L2), not a
   lumped diagonal.  On the Schur complement the floor shifts S by
   +eps·M: at depth the floored Schur is SPD with kappa ~ 1.5-2
   (L2: [6.5e-3, 1.0e-2]) — the 1e-2 floor DOMINATES the latent block
   at depth, which is exactly why the Schur PC's S-block solve is
   trivial and why the sqrt(kappa)·eps stagnation mechanism exists
   (the PC inverts a system differing from J by O(eps) on the
   degenerate subspace; harmless for Newton, which stays LU-exact).

**Status of the scalability claim.**  Concept proven: a matfree operator
with a Schur-fieldsplit preconditioner on the assembled floored Pmat
reproduces LU exactly at all tested depths, with the degeneracy handled
entirely inside the preconditioner.  Scaling limit reached at ~8k dofs by
outer-GMRES growth from the diagonal-Schur approximation; the fix
(better Sp) is classical preconditioner engineering, and the spectral
data above quantifies the target (capture B^T K^{-1} B more faithfully
in the transition band where the unfloored spectrum crosses zero).

**Sp-upgrade verdict (per-cell block Schur, measured 2026-09-08).**  The
Finding-6 attribution of the outer-GMRES growth to Schur-approximation
quality is REFUTED by direct experiment
(`experiments/schur_percell.py`, one new file, no existing files modified):
replacing the diagonal Sp by the exact per-cell broken-basis Schur
complement  Sp = D_floor - (1/alpha) C0,
C0 = sum_cells Lm_c^T (alpha ks_c)^+ Lm_c  (per-cell 3x3 Moore-Penrose
pseudoinverse; constant mode zero-coupled in the "pinv" variant,
mass-regularized tau = 1e-2 in the "reg" variant), attached PETSc-natively
via `setFieldSplitSchurPreType(USER, Sp)` and refreshed per SNES solve,
leaves every solve BIT-IDENTICAL to the baseline (prox/newton/err counts
unchanged at L0/L1/L2).  Measured coupling scale: |C0| ~ 7.3e-4 x |D_floor|
-- the coupling term B^T Khat^{-1} B is O(h^4) against D's O(h^2),
structurally subdominant at every h, unlike the divergence-coupled saddle
systems the hierarchical-pG recipe targets.  The growth driver is therefore
the floored-D / degenerate-row mechanics of the coupled system (the
eps-floor-dominated latent block), consistent with spectrum-anatomy point
4; the next lever is the floor treatment itself (active-set /
pseudoinverse rows), not the Schur coupling.

**Growth-driver decomposition (measured 2026-09-09,
`experiments/growth_driver.py`; parent fixed the agent's petsc4py
context-manager `getArray` bug that had killed the agent's runs —
plain `getArray(readonly=True).copy()` + `setArray()` is the working
idiom in this petsc4py).**  Four right-preconditioners applied to the
SAME monolithic J at the SAME deep converged states (captured from an
actual max-its Newton solve), outer GMRES rtol 1e-6 restart 250:

| PC variant | L0 | L1 | L2 |
|---|---|---|---|
| (a) exact block-LU: exact K^-1 + exact UNfloored S^-1 | **1** | **1** | **1** |
| (b) exact block-LU with FLOORED S (selfp+floor analogue) | 145 | 895 | 1000 (DIVERGED_ITS, rel-resid 1.71) |
| (c) active-set identity rows on dead latent dofs | 825 | 178 | 1000 (DIVERGED_ITS) |
| (d) baseline diagonal-Sp reference | 123 | 505 | 1000 (DIVERGED_ITS) |

State facts: dead rows 45/177/765 (|d|<1e-8), phi_min -425/-763/-318,
captured RHS from the max-its solve, ||b|| ~ 1e-5.

**Verdict.**  (1) The BEST-POSSIBLE PC is perfect at every depth: exact
K^{-1} + exact unfloored S^{-1} gives GMRES its = 1 flat across levels.
The growth is therefore 100% PC-fixable — it is NOT an operator property,
and deflation of the dead subspace is NOT required for the solve itself.
(2) The driver is the FLOOR: the floored exact-S variant reproduces the
entire growth curve (145/895/1000 vs baseline 123/505/1000-scale) — the
eps·M floor on the latent block is what degrades the PC at depth, exactly
the O(eps)-mismatch-on-the-degenerate-subspace mechanism of
spectrum-anatomy point 4.  (3) The naive active-set identity-row
treatment is WORSE than the floor at L0 (825 its) and diverges at L2:
zeroing the dead rows' diagonal while keeping their coupling columns
destroys the Schur balance the floor was preserving — the dead rows must
be eliminated consistently with the coupling, not just patched.  (4)
Practical consequence: the scalable path is a *consistent* treatment of
the degenerate latent block (e.g. solve the floored system but correct
the PC's action on the dead subspace, or decay eps with alpha), NOT
deflation.  Note the per-variant rel-resid column of the current harness
is mis-scaled (computes ||Jx||/||b|| instead of ||Jx-b||/||b||); the its
and converged-reason columns are the valid measurements.

**Rank-k dead-subspace correction (measured 2026-09-09,
`experiments/rankk_correction.py`; L1 verified independently by the
parent).**  The candidate fix — correct the floored PC on span(V), V =
latent unit vectors on dead dofs, so that the corrected PC is exact there
(like variant (a)) while keeping the safe floored action elsewhere — is
REFUTED.  Verified honest: correction exact on span(V) to rel 1e-17;
raw V^T J V effectively singular (cond 9e16..8e59) and never inverted;
the k x k Schur-Gram V^T S_a V replacing it is benign (cond 1.7e3..2e4);
correction-on/off changes its counts as expected (145->11 at L0).  But:

| variant | L0 | L1 | L2 |
|---|---|---|---|
| (b) floored exact-S | 145 | 895 | 1000 div (rel 1.47) |
| rank-k on span(V), k=45/177/765 | **11** | 1000 div (rel 6.2e-3) | 1000 div (rel 0.38) |
| rank-k on eig-band, k+64/240/829 | **9** | 1000 div (3.9e-4) | 1000 div (0.16) |
| (a) exact unfloored | 1 | 1 | 1 |

**Why it fails (measured deficiency diagnostic).**  The floored PC's
deficiency E - I, E = D S_f^{-1} (latent block of the preconditioned
operator), has SLOWLY-DECAYING singular values (sigma_max ~ 1e2/2e2/3e1)
whose band above 0.1 has rank >= 70/241/829 — LARGER than the strictly-
dead count k = 45/177/765, with sigma_k still O(1) (0.998/0.974/0.85).
The eps-floor therefore damages a whole TRANSITION BAND of the latent
block (rows where D is small but not dead), not just the strictly-dead
rows; correcting any finite sub-band leaves a long stagnating tail.
Consistent with variant (c)'s failure: inconsistent partial patching of
the latent block loses to the consistent floor.

**Implication.**  Residual growth is NOT low-rank-curable.  The scalable
treatments left standing: (i) reproduce the exact unfloored Schur action
across the band (variant-(a) structure — matfree S application, e.g.
inner CG on -S, S SPD up to sign, h^2-scaled, per-cell Sp as inner PC;
machinery exists in schur_percell.py), or (ii) decay eps with alpha so
the floor's damage band shrinks with the proximal step.  The Stage-2
promotion machinery (PCShell wrapper, per-prox refresh, correct getArray
idiom) is included in the file for whichever variant succeeds.

**RESOLVED — floorless Schur configuration (measured 2026-09-09,
`experiments/eps_schedule.py`; psi_floor=0 end-to-end verified
independently by the parent).**  The psi_floor is UNNECESSARY on the
schur/upper/selfp config: Sp = D - B diag(K)^{-1} B^T is nonsingular
without any floor (dead rows of D are carried by the coupling term; Sp
LU-solve residual ~3e-16), and the floor was load-bearing ONLY for the
additive/diag-inversion path (Finding 1's floor sweep).  End-to-end with
psi_floor=0, levels 0-2 (parent verification):

| level | prox | newton | err(u_h) | wall | max outer its |
|---|---|---|---|---|---|
| 0 | 8 | 21 | 1.553e-02 | 0.9 s | 14 |
| 1 | 2113 | 11 | 23 | 3.599e-03 | 2.0 s | 29 |
| 2 | 8321 | 8 | 19 | 8.375e-04 | 6.5 s | 48 |

The outer-GMRES growth is GONE as a limiter: 14/29/48 vs the floored
baseline 45/130-560/400-926 (L2 wall 5.5-6.5 s vs 81 s).  Newton stays
exact (floor was Jp-only), errors LU-identical.  The safety ladder is a
BAND, not a cliff: eps <= 1e-5 is safe at every level (indistinguishable
from eps=0), eps=1e-4 is already the knife edge at depth (540-650 its,
non-monotone — the transition band), eps >= 1e-3 kills L2 cold solves.
The drift-concentrated floor (contact-scale ~1e-3 via psi_floor_drift)
lands in that bad intermediate band and is REJECTED for the Schur config
(diverged cold at all levels; full solves converge LU-identically but
with max-its 275/326/353).  This RETRACTS the floor's necessity for the
Schur path: Finding 1's "floor is necessary and correct" stands for the
additive config only; on schur-upper-selfp the floor is pure cost, and
`psi_floor: 0` is the recommended setting.  Residual growth 14/29/48 is
mild and no longer near any guard; whether it stays bounded at levels
3-4 (with CG+GAMG replacing MUMPS on K) is the next scalability datum.





---

## hpG port (2026-09-10, stage 0/1 gates PASSED; stages 2/3 implemented,
pending promotion)

Spec: `HPG_PORT_SPEC.md`; results of record:
`experiments/hpg/RESULTS.md` (all headline numbers verified by parent
stage0_gate/stage1_machinery reruns).  The faithful Firedrake port of
Papadopoulos's hierarchical-pG obstacle solver is UP: quad meshes, u in
CG_p (hats + Jacobi bubbles = same polynomial space), latent psi in
DQ_{p-2} spectral (Legendre modal, nodal dual at Gauss points),
psi_spaces route with NO lvpp.py edits.

Stage-0 (machine-precision): Lemma-3.2 psi-mass modal-diagonal to 4e-18
(exact — needs the discovered modal<->nodal Vandermonde; raw Function.dat
is nodal, not modal); D_psi exactly cell-block-diagonal (off-block 0.0);
Y = P_n - P_{n+2} basis: 1D stiffness diagonal but 2D cell Ahat has 4
parity blocks (the paper's "4N blocks") — closed-form matches 9e-14.
Direct control vs analytic sphere solution: L0 p=2 (1345 dofs) err
1.19e-3 (our P1: 1.55e-2 at 545); L1 p=2 (5249) 1.04e-4 (our P1: 3.60e-3
at 2113) — **~13-35x better accuracy per dof, the paper's selling point
confirmed on our benchmark**.

Stage-1: alpha-independence verified 3.6e-15 (the cached-Cholesky trick);
Shat SPD at the deep state (min eig 8.5e-7 at beta=0); cold true-Schur
inner GMRES with Shat PC: 14/12 its (shallow/deep) — polylog claim
consistent at p=2, 16x16.

Stages 2/3 PROMOTED (2026-09-10, `experiments/hpg/promote_hpg.py`; graded
g4 row verified by parent rerun): uniform L0-L2 p=2 — two-stage P_F matches
the serial-direct controls bit-consistently (prox 7-8, newton 20-23, err
1.192e-3/1.036e-4/2.651e-5) with outer FGMRES max 2 its FLAT (vs our P1
selfp 14/29/48, vs the paper's 14-36) and inner its/apply 12.8→27.0
(polylog-consistent).

Stages 2/3 PROMOTED (2026-09-10, `experiments/hpg/promote_hpg.py`; graded
g4 row verified by parent rerun): uniform L0-L2 p=2 — two-stage P_F matches
the serial-direct controls bit-consistently (prox 7-8, newton 20-23, err
1.192e-3/1.036e-4/2.651e-5) with outer FGMRES max 2 its FLAT (vs our P1
selfp 14/29/48, vs the paper's 14-36) and inner its/apply 12.8→27.0
(polylog-consistent).

**TRUE-MATFREE hpG (2026-09-10, `experiments/hpg/matfree_hpG.py`; smoke
verified by parent).**  The solve path is now free of global
factorizations: K^-1 = capped CG+GAMG/hypre (AMG built once per mesh as
preconditioner-only), true-S matvec via the triple product, Shat PC =
batched CELLWISE Cholesky, E_beta = 0 in the operator, capped inner GMRES
(rtol 1e-4, cap 40), outer FGMRES.  Correctness: bit-consistent with
serial-direct at p=2 (L0 8/22/1.192e-3 identical).  The fair high-order
comparison (L0 p4, same capped config, both arms): assembled+cached
8/30/9.542e-5/117 s vs matfree 8/30/9.542e-5/404 s — IDENTICAL
prox/newton/err/outer-its; the matfree cost is purely per-apply A-CG time
(3.5x wall at p4, 1.3x at p2; worsens with kappa(A)).  Graded 51.4x p4
(37k+20k dofs): assembled arm HOLDS (8/22, 1.568e-5, 535 s); matfree arm
— two honest negatives: GAMG STALLS on graded p=4 K0 (1994/1994 nonconv);
hypre BoomerAMG healthy per-iteration (28.7 its avg, zero nonconv) but
projects to ~2000-2500 s, beyond budget.  **The open preconditioner
question for truly-matfree hpG is the AMG for graded high-order K; the
iteration side is fully matfree-clean.**  beta sweep: all {0,1e-5,1e-4,1e-3} identical →
**floorless TRANSFERS to hpG** (beta_Shat=0 viable; 1e-3 if anything
worse).  **Graded chain: hpG Shat SURVIVES grading — clean solves at
26.4x/55.3x/114.8x/226.1x (up to 25.7k dofs), both published and capped
arms, outer max 2 its, NO divergence (our selfp P1 diverged at 14.2k)** —
the Remark 6.1 gap closes in hpG's favor.  p=4 honest negative for the
published arm (inner GMRES wedges at the 500-cap with reason −3 once α
deepens; >20 min killed); OUR capped-inexact discipline rescues it (8/30,
err 9.542e-5 vs direct 9.620e-5, 117 s).  Wall: hpG two-stage
4.0/14.2/96.9 s vs direct 0.3/0.4/1.5 s (P_F costs ~30-60x a direct solve
at L2; the paper's fast transforms would close this) — iteration counts,
not wall, are hpG's win.

## Not yet demonstrated (do not overclaim)

- QVI without continuation (`remap` argument is structural, porous example
  not run through LVPP).
- Parallel correctness/performance of the mixed solve.
- Any effectivity/reliability theory for the saddle estimator.
- The SNES-nonlinear-preconditioner framing (LVPP as a custom PETSc SNES
  type / TAO-style wrapper) — design sketched in session notes; the
  preconditioner-quality measurement per family is an open experiment.
- Generalization beyond box+gradient constraint families through the
  solver (Hellinger proven in the intersecting test; simplex/Signorini/
  eigenvalue are interface-complete but unexercised).

---

## Next experiments

1. **eps(x) floor — DONE (2026-09-04); full measured envelope + verdict in
   Finding 1.**
2. **Schur fieldsplit at depth + S-spectrum anatomy + Sp-upgrade/rank-k/
   eps-schedule decomposition — DONE (2026-09-08/09); Finding 6.**  Final
   verdict: the eps floor is UNNECESSARY on the Schur config and was the
   entire growth driver — psi_floor=0 gives LU-identical solves with
   max outer-its 14/29/48 (floored baseline: 45/130-560/400-926).  The
   Schur config is now the recommended production configuration.  Next
   levers: levels 3-4 scaling with CG+GAMG on K replacing MUMPS, MPI
   smoke, then the AMR-vs-uniform preconditioner-health test (dead rows
   37/177/749 prediction) as the bridge to the viamr story.

### Scaling workstream (launched 2026-09-09, parallel agents)

Three parallel agents on the floorless Schur config
(`pc_fieldsplit_type schur, upper, use_amat False, selfp, fieldsplit_0
preonly+lu(MUMPS), fieldsplit_1 preonly+lu(MUMPS), psi_floor 0`):

1. **ScaleUp** (levels 3-4 = 33k/132k dofs + CG+GAMG-on-K variant +
   deep-state floor regression): MeshHierarchy(base, 4) exceeds memory
   for serial MUMPS on this box? — measure.  Watch: K-block LU cost
   growth, outer-its curve 14/29/48 -> ?, and the Finding-5 saturation
   corner (|psi| ~ 1351 killed MUMPS once; the floorless config's dead
   rows are handled by Sp, but fieldsplit_0 still factors K).
2. **AMRHealth** (bridge to viamr): viamr-marked adaptive mesh vs
   uniform at matched dof budget — count dead rows and outer-GMRES
   its on the same floorless config.  Prediction: dead rows
   O(perimeter/h) not O(area/h^2), flatter outer-its curve.
3. **ParSmoke** (mpi smoke): 2-rank run of the floorless Schur config
   on the existing sphere harness (`@pytest.mark.parallel(2)` style or
   a standalone mpiexec driver) — correctness (LU-identical counts) +
   scaling of the two exact MUMPS block solves.

Success criteria: L3/L4 solve LU-identical (or deltas explained) with
outer-its not regressing past the guard; AMR at matched dofs shows
fewer dead rows and flatter its than uniform; parallel run correct and
wall-improving.

### Scaling workstream results (2026-09-09; all headline numbers
verified by parent reruns)

**1. Levels 3-4 (`experiments/scale_up.py`; L3 verified by parent).**
Both solve.  L3 (33025/field): 8/20, err 1.905e-04, max-its 105, wall
~42-55 s, MUMPS 20% of wall, RSS 756 MiB.  L4 (131585/field): 8/21,
err 5.006e-05, max-its 329 (late prox steps 301-329, beyond restart
250; convergence still RTOL), wall ~532 s, MUMPS 74 s (14%), RSS 1737
MiB.  Errors clean O(h^2) across L0-L4 (1.553e-2/3.599e-3/8.375e-4/
1.905e-4/5.006e-5); psi depth stable (-318..-372, no runaway).  Dead
rows follow the perimeter-h law ~4x per level: 749/3073/12437 (<1e-12,
L2/L3/L4).  **Honest negative: outer-its growth is polynomial
(~3.1x per 4x-dofs level, roughly N^0.8), NOT flat** — 14/29/48/105/329.
L5+ needs a better S-PC; the floorless config still dominates the old
floored baseline by an order of magnitude at every size.

**2. CG+GAMG on K: correct but a wall LOSER.**  LU-identical Newton at
L0/L1 (rtol 1e-8), +1 Newton at L2 (rtol 1e-10 restores 8/19 exactly),
err unchanged at every level.  But wall penalty 1.6x/2.6x/2.2x/3.8x and
WORSENING with depth (L3: 206.7 s vs 54.8 s).  Measured mechanism: the
selfp/upper Schur apply invokes a fieldsplit_0 KSP solve on EVERY outer
GMRES iteration (1085 solves at L3, ~128 ms each; PCApply = 75% of
wall) vs one triangular-solve pair for preonly+MUMPS (~23 ms/apply).
Inner CG itself is healthy (12-14 its/solve, flat in depth).  Verdict:
keep MUMPS until its cost dominates (L4: 74 s of 532 s — not yet);
revisit GAMG only when MUMPS wall share >50% or memory binds.

**3. MPI smoke (2 ranks): correctness PASS, performance neutral.**
L0/L1 LU-identical at 2 ranks (8/21/1.553e-02, 11/23/3.599e-03),
feasible, curves agree with serial to +-1 its (MUMPS pivot roundoff).
`pc_fieldsplit_use_amat: True` also WORKS in parallel on this install
(LU-identical both settings) — the flagged risk area did not
materialize.  Wall: no gain at these sizes (1057/4226 dofs/field;
distribution overhead > factorization savings) — defer the parallel
scaling datum to L2+ where MUMPS cost pays.  Harness: mpiexec -n 2
experiments/par_smoke.py.

**4. AMR-health prediction REFUTED (measured; band arm verified by
parent).**  Finding 1's "free-boundary-aware AMR is the degeneracy-
minimizing strategy" is WRONG for the preconditioner on this problem:
at matched dofs, band-refined (free-boundary ring only, viamr SBR)
6861-dof mesh has 5169 dead rows (uniform L2 8321-dof: 765 — 6.8x),
max outer its 323 vs 48 (6.7x), and the band chain BREAKS DOWN
entirely at 14245 dofs (ksp=-5) where uniform sails through.  udobr
(standard viamr pipeline, 11197 dofs): 2565 dead rows, 281 max-its.
Mechanism: **the refined free-boundary ring is a dead-row FACTORY** —
sharpening the boundary converts smeared compromise nodes into
cleanly-pinned contact nodes; ring dead rows grow at O(perimeter/h_ring)
and h_ring shrinks every level, so the dead FRACTION of latent dofs
explodes 7% -> 50% on band meshes (uniform holds ~9%).  Second
mechanism: outer-its blowup correlates with grading strength
h_max/h_min (1x:48; 10x:78-275; 45x:323) — the selfp Sp quality
degrades under grading via the diag(K)^{-1} mismatch amplified by
(h_max/h_min)^2.  Also a MEASUREMENT correction: the raw
|diag(D)|<1e-8 census is h-unfair on graded meshes (D_ii ~ h^2, refined
cells cross the fixed threshold early; band_l6 raw 12261 vs h-fair
4052) — the h-fair census is psi < log(tol); on uniform meshes the two
agree exactly.  **Consequence: uniform refinement is simultaneously the
dead-row- and its-MINIMIZING mesh at matched dofs; the AMR link in
Finding 1's "Relation to free-boundary-aware meshes" is retracted for
the preconditioner (it survives only for ACCURACY per dof, Finding 3's
estimator — the band mesh DOES converge 4.957e-3 at 6861 dofs vs uniform
L1 3.599e-3 at 2113).  The paper's bridge between viamr and the solver
must be reframed: AMR buys accuracy per dof, and the solver cost of
adaptivity is the grading-sensitivity of the selfp Schur PC — itself a
finding (a boundary-layer-aware Sp is the open fix).**

**Net scaling verdict.**  The floorless Schur config solves to
LU-quality through 132k dofs serial on one box, with polynomial
(roughly N^0.8) outer-its growth and no memory wall.  Remaining
blockers before any "scalable" claim: the N^0.8 outer-its growth
(needs a grading-robust S-PC or matfree exact-S apply), MUMPS serial
factorization at L4+ (CG+GAMG measured as a wall loser for now), and
the AMR-solver interaction which is real but opposite in sign to the
old prediction.

**Grading-robust S-PC experiment (2026-09-09, `experiments/grade_pc.py`;
variant table verified by parent rerun).**  Three candidate fixes tested
on the SAME extracted graded states (band_l4 22.8x, band_l5 45.6x
grading) with the same captured max-its RHS; uniform L2 as control
(cold harness reproduces production exactly there: 8/19, 8.375e-04,
maxits 48):

| variant | uni_l2 | band_l4 (22.8x) | band_l5 (45.6x) |
|---|---|---|---|
| base (selfp Sp, exact block-UL harness) | 22 | 39 | 51 |
| gold (a) exact unfloored S | 1 | 1 | 1 |
| A-pinv / A-reg (graded-aware diag Sp) | 89 / 92 | 85 / 85 | 90 / 90 |
| **B inner-CG on -S, cap 40, rtol 1e-4** | **6** | **10** | **15** |
| B rtol 1e-3 | 5 | 12 | 21 |
| B rtol 1e-2 | 8 | DIVERGED_BREAKDOWN | DIVERGED_BREAKDOWN |
| C per-cell exact Schur coupling | 29 | 53 | 83 |

**Diagnosis: the diag(K)^{-1} mismatch hypothesis (Scaling-workstream
mechanism 2) is REFUTED.**  kappa(diag(K)^{-1} K) grows only 3.3e3 ->
4.4e3 -> 8.9e3 across h-ratios 1x -> 45.6x (the hypothesis predicted
(h_max/h_min)^2 ~ 2000x), and ||Sp - S||_2/||S||_2 stays ~1.6e-3 on
graded meshes (~10x worse than uniform's 1.6e-4 — degraded but still an
excellent approximation).  Variant A (built on the refuted hypothesis)
is an honest negative: WORSE than baseline on every mesh and it worsens
the Sp-S norm (5.7e-3).  Variant C (per-cell exact Schur coupling on
graded meshes, schur_percell machinery) is also negative — extends the
uniform-mesh per-cell refutation to graded meshes.  **Variant B is the
winner and the only fix that helps: matfree true-S inner CG (S v = D v -
B^T(K^{-1}(B v)), one sparse-LU K solve per matvec, Jacobi-scaled, cap
40) at inner rtol 1e-4 gives 6/10/15 its — BEATS selfp at every state
and stays nearly flat under 45.6x grading (15 at band_l5 < uniform's
22).**  Inner rtol 1e-2 is a graded-mesh knife-edge (outer breakdown at
both band states; inner CG must be <= 1e-4).  gold(a) = 1/1/1 on graded
meshes confirms the graded growth is 100% PC-fixable.  Honest caveat:
the cold harness's base PC UNDER-reproduces graded production max-its
(39/51 vs 297/323 — production's upper-factorization apply differs from
the harness block-UL), so promotion magnitudes need end-to-end runs;
the variant comparisons are same-state/same-RHS and internally
consistent.  **Promotion queue: variant B (rtol 1e-4, cap 40) to full
solves on the graded chain — the remaining open promotion.**

**End-to-end promotion result (2026-09-10, `experiments/promob.py`;
uniform L2 cross-checked with the inner-gmres arm by the parent).**
Route: monolithic top-level PYTHON PC replicating upper-factorization
semantics (petsc4py instantiates pc_python_type contexts with no
constructor args, so fieldsplit_1 sub-PC PYTHON cannot receive the lvpp
handle; outer FGMRES rtol 1e-6).  Results:

- Uniform control: LU-identical everywhere — L0 8/21/1.553e-02, L1
  11/23/3.599e-03, L2 8/19/8.365e-04 (LU 8.375e-04; 4th-digit noise).
- Graded band chain: band_l4 161 max-its vs selfp 297 (deep proximal
  steps 297 -> 4-5); inner-GMRES arm solves band_l5 (8/17, err
  4.995e-3, maxits 188 vs selfp 323) and band_l6 (14245 dofs) records
  EVERY linear solve CONVERGED_RTOL (2-192 its) where selfp production
  DIVERGED ksp=-5 — the graded-mesh divergence is eliminated.
- Costs, measured honestly: variant B is ~4.5x slower than selfp on
  UNIFORM meshes (the first, coldest Newton solve pays 62-70 its at
  every level; s_wall 22 s of 28.9 s at uni_l2) — its value is
  grading-robustness, not uniform speed.  With inner CG capped at 40,
  band_l5's proximal iteration STALLED (increment 4.243e-4 plateau,
  all linear solves 4-5 its): the capped inexact Schur action
  perturbs the Newton matrix and the proximal map drifts within
  tolerance; the identical chain with inner GMRES (looser effective
  cap, restarts counted) solves band_l5 in 8 prox — the stall is
  CG-specific, not mesh/implementation-driven.
- Generality: the S action is plain matvecs (D v - B^T K^-1 (B v)) with
  no symmetric assumption; the inner method swaps CG -> GMRES unchanged
  in the same code (verified end-to-end), as required for
  non-potential LVPP families (nonsymmetric/indefinite S).

**Final production recommendation.**  Uniform meshes: selfp Schur
(psi_floor=0) — fastest.  Graded/adaptive meshes or stalled proximal
steps: inner GMRES on -S (rtol 1e-4, restart ~40) — kills the graded
blowup and the l6 divergence, family-general, at a measured wall
premium (or use only when selfp stalls).


3. NSV03 dual-aware restructuring (Finding 4.2).
4. Saddle-estimator reliability: compare eta against the inactive-set error
   decomposition (global L2 minus the active-set interpolation floor).
5. Porous QVI through `remap` (viamr/examples/porous.py as the template).
6. Parallel mixed solve smoke (`@pytest.mark.parallel(2)`).
---

## Paper gap analysis & verdict (2026-09-08)

**Verdict: arXiv-note / workshop grade, NOT yet a SISC/CMAME submission.**
The material is a diagnosis + one validated configuration on one problem
(sphere), serial, with a *growing* outer-GMRES curve and no theorem. Six
gaps, ranked by reviewer damage:

1. **Scaling evidence (fatal without it).** Three levels ending at 8,321
   dofs. UPDATE (2026-09-09): the outer-GMRES growth question is RESOLVED
   (psi_floor=0 on the Schur config; max its 14/29/48, Finding 6), so the
   remaining scaling work is: levels 3-4 (33k, 132k), CG+GAMG on K
   replacing MUMPS (itself a serial wall), and an MPI smoke.
2. **The AMR experiment — now RUN (2026-09-09), and the old prediction
   is REFUTED** (see "Scaling workstream results" under Next
   experiments): at matched dofs AMR has MORE dead rows and a WORSE
   outer-GMRES curve; uniform is the solver-optimal mesh.  The
   differentiator must be reframed: AMR buys accuracy per dof (Finding 3
   estimator, validated), the solver cost of adaptivity is the measured
   grading-sensitivity of the selfp Schur PC, and a grading-robust S-PC
   is the open fix that would reconnect the two halves positively.
3. **A theorem-grade core (cheap).** Provable with generalized-eigenvalue
   mass-stiffness bounds: S = D - B^T K^{-1} B satisfies S <= -c h^2 I
   uniformly in the degeneracy (Finding 6.3 becomes a proposition); the
   dead-row count equals the measure of the active set. One week.
4. **Floor-role correction owed (Finding 6.4's caveat).** The floored-
   Schur kappa ~ 1.5-2 measurement used the diagonal D + eps I
   approximation; the actual lvpp floor form is a FULL mass matrix and
   true floor eigenvalues are Theta(h^d eps) — orders below eps — so the
   genuinely floored Schur at depth is likely still indefinite and "the
   floor dominates the latent block" is true only relative to D.
   Afternoon: rerun `spectrum_anatomy.py` with the actual floor form.
   Then: does schur-upper-selfp need `psi_floor` at all (its D-block is
   never inverted)?
5. **Second problem class.** Porous QVI through `remap` (constraint
   depends on the solution — the natural stress test), or Signorini
   (structurally different boundary family). One, minimum.
6. **Competitor table.** PETSc `vinewtonrsls` active-set (already in
   PETSc) and IPOPT: wall time, feasibility, robustness at depth vs
   LVPP-matfree and LVPP-direct. The row-mutation foil the motivation
   rests on is currently asserted, not benchmarked.

**Two shapes.** (A) solver paper — "matrix-free solvers for latent-
variable proximal Galerkin systems": exactness, mechanism + propositions
(3), Schur PC, Sp restoration, scaling study (1). Incremental; competes
with hierarchical-pG on their turf. (B) merged claim — "contact-aware
adaptivity makes latent-variable VI solvers tractable": AMR shrinks the
degenerate subspace (2), restores Krylov health, improves a posteriori
accuracy (Finding 3). The only framing where the contribution is not
following Papadopoulos. **Recommendation: (B) with (A)'s material as the
solver core.**

**Critical path (updated 2026-09-09).** Sp-upgrade/floor question RESOLVED
(psi_floor=0); remaining: scaling + CG/GAMG -> AMR experiment -> QVI ->
competitor table -> propositions (gap 3). The floor-role rerun (old gap 4)
was partially answered by the eps-sweep safety ladder; the TRUE-floor
(eps*M, not D+eps*I) spectrum rerun of Finding 6.4 is still owed.

### Experiment results: eps floor sweep (2026-09-03/04, final)

Script: `lvpp/experiments/matfree_fieldsplit.py`.  Script hygiene
discovered this session (now built in): PETSc options labels must be
sanitized before use as options_prefix ("=" or spaces in the prefix
cross-contaminate the options database and produce phantom
nondeterminism — tags are now `[A-Za-z0-9_]` only); failed configs print
SNES/KSP converged reasons (ksp=-5 breakdown -> snes=-3 linesearch is
the degeneracy signature); selector modes: `all|eps0|drift|probe`.

Configs: LU baseline (Jp = J); matfree + fieldsplit with constant floor
eps0 in {1e-3, 1e-2, 3e-2, 1e-1}; drift-scaled floor
eps(x) = eps0 + c·alpha·|lambda(x)| for c in {0.03, 0.1, 0.3, 1.0}
(eps0 = 1e-3); inexact-Newton (ksp_rtol 1e-2), Eisenstat-Walker, Schur
fieldsplit, and operator-floor variants.  Full table + verdict in
**Finding 1 ("Measured")** above.

- LU baseline errors: level 0: 1.553e-2, level 1: 3.599e-3, level 2:
  8.375e-4 (prox 8/11/8).
- Level 0: eps0 in {1e-2, 3e-2, 1e-1} converge with err identical to the
  LU baseline (7 prox / 22 Newton); drift floor c = 0.03 also converges.
  eps0 = 1e-3 fails; drift c >= 0.1 fails (floor overshoots the early
  proximal steps where alpha·|lambda| is momentarily large).
- Levels 1-2 with tight inner solves (rtol 1e-6): every preconditioner
  floor fails via KSP_DIVERGED_BREAKDOWN -> SNES_DIVERGED_LINE_SEARCH.
  Mechanism (measured): floating-point stagnation at the
  sqrt(kappa)·eps attainable-accuracy floor set by the deep-latent
  near-null cluster (Finding 1 verdict + scaling-falsification blocks);
  restart-independent; no Jp-only floor can help.
- Inexact Newton (fixed ksp_rtol 1e-2) + floor 1e-2: level 2 converges
  (8/65, err 8.376e-04 = LU-exact); level 1 is a documented knife-edge
  (fails at every tested setting including EW).
- With `ksp_type: preonly` + LU, ANY eps0 > 0 on Jp breaks the solve:
  with `ksp_type: preonly` the PC matrix *is* the solve matrix, so a
  floored Jp changes the system rather than preconditioning it.  The
  floor therefore requires a Krylov inner solve — also the libCEED-
  relevant configuration.
- Operator-level floor (`psi_floor_operator=True`): fails at EVERY level
  including level 0 — the proximal dynamics need exact Newton steps;
  direct evidence for the Jp-only design rule.
- History note: the earlier "refined-mesh fieldsplit block-extraction"
  hypothesis (09-03) is retracted — that failure signature
  (DIVERGED_PC_FAILED at iteration 0, including on level 0) was the
  missing `pc_fieldsplit_use_amat: False` flag, not a mesh-refinement
  issue.  With the flag, the fieldsplit PC sets up and iterates at all
  levels.

**Scaling/equilibration test (2026-09-04, refuted).**  The candidate fix
"two-sided diagonal scaling delta-phi = e^{-phi/2} w" (IPM-style slack
scaling) was implemented in its cheapest exact form: right-preconditioned
GMRES with pointwise Jacobi (ksp_pc_side right, pc_type jacobi, no floor,
no fieldsplit) — the Krylov operator J diag(J)^{-1} is similar to the
congruence diag(e^{-phi/2}) J diag(e^{-phi/2}).  Result: FAILS at every
level including level 0 (KSP_DIVERGED_BREAKDOWN, residual perfectly
stagnant across Arnoldi iterations).  Matrix-level inspection at a
proximal iterate (phi >= -32.6) explains why and CORRECTS the mechanism:

- the latent diagonal is not purely e^phi-scaled: it spans 1e-8..5e-2
  with an O(1/alpha)-like healthy bulk and a degenerate tail;
- the u-block response to a dead-subspace latent perturbation is
  O(1e-2) per unit — the coupling B does NOT fade with e^phi;
- hence the near-null directions MIX both blocks
  (delta_u ~ -K^{-1} B delta_phi), and no diagonal (left, right, or
  two-sided) scaling can decouple them.  (Correction from the later
  spectral anatomy, Finding 6: S itself never collapses — lambda_S_min
  is h²-scaled and invariant; the near-null space is the monolithic
  D-rows' flat manifold.  The two-block mixing statement stands; the
  "S collapsing" wording was wrong.)

Consequence for remedies: any PC-based fix must capture the COUPLING
(a true Schur-complement approximation, or a saddle-aware transformation),
not just diagonals.  Live paths: (i) the validated inexact-Newton recipe
(fixed loose ksp_rtol; LU-exact errors at levels 0 and 2), (ii) a Schur
fieldsplit that actually sets up (earlier attempts died at PC setup and
need a dedicated session), (iii) a saddle-aware variable transformation
(couples both blocks; forms-level work).

---

## Finding 7: Synthesis with Hierarchical Proximal Galerkin (Papadopoulos 2026) & Cross-Framework Architecture

See full technical specification in `lvpp/HIERARCHICAL_PG_INTEGRATION.md`.

1. **Convergence of Architecture**: Both hpG (high-order tensor-product p-FEM, Julia) and `lvpp` (low-order unstructured FEM, Firedrake) converge on the outer-FGMRES / inner-Schur factorization paradigm ($P_F$) applying $S = D_\psi - B^T A_\alpha^{-1} B$ matrix-free.
2. **Adaptive Benchmark Gap in hpG**: In `hierarchicalPG.pdf`, all $h$- and $hp$-adaptive runs (Figure 6, Figure 7) were computed using **sparse direct factorizations** (Remark 6.1). The iterative Schur preconditioner was benchmarked only on uniform meshes.
3. **Graded Mesh Mechanism**: On graded meshes ($h_{\max}/h_{\min} \gg 1$), decoupled Schur preconditioners degrade due to the $O((h_{\max}/h_{\min})^2)$ stiffness scale disparity across refinement interfaces and the "dead-row factory" phenomenon along refined contact boundaries.
4. **Resolution via `lvpp` Discoveries**:
   - **Inner PCG on $-S$**: Since $-S = -D_\psi + B^T A_\alpha^{-1} B$ is strictly SPD, inner GMRES in hpG can be replaced with Preconditioned CG, reducing memory and work to a 3-term recurrence.
   - **$J$ vs. $J_p$ Decoupling**: Keeping the matrix-free action of $S$ exact while placing stabilization $E_\beta$ or contact floors $\epsilon(x)$ strictly on the preconditioner $\hat{S}$ prevents the solution error plateaus observed in Table 2 of the paper without requiring $\beta \to 0$.
   - **Inexact Capping**: Pairing outer FGMRES with an inexact, capped inner PCG solve ($\text{rtol} = 10^{-4}$, max 30 iterations) bounds outer iterations to 10-15 even under severe ($45.6\times$) mesh grading.

---

## Finding 8: Strategic Direction: Partitioned Multiphysics Proximal Galerkin

See full literature review and architectural framework in `lvpp/PARTITIONED_MULTIPHYSICS_PROXIMAL_GALERKIN.md`.

1. **Open Literature Gap**: Confirmed that no published literature currently incorporates Proximal Galerkin or LVPP into partitioned multiphysics coupling frameworks (preCICE, FSI with contact, poromechanics, coupled fracture).
2. **The Root Bottleneck**: Standard partitioned multi-secant / quasi-Newton accelerators (IQN-ILS, Anderson acceleration) assume $C^1$-smooth interface response maps. Active-set / penalty contact methods introduce non-smooth ($C^0$ / discontinuous) derivative jumps that cause chattering and divergence; penalty stiffness worsens FSI added-mass instability.
3. **Four Architectural Paradigms**:
   - **Paradigm 1 (Subdomain-Level PG, preCICE-Ready)**: Constrained participants solve via PG, smoothing the DtN interface map to $C^1$ and restoring theoretical superlinear convergence to black-box quasi-Newton accelerators.
   - **Paradigm 2 (Interface LVPP / Proximal Robin)**: Replaces non-smooth Signorini interface conditions with smooth Proximal Robin–Neumann conditions parameterized by an interface latent field $\psi_\Gamma \in L^\infty(\Gamma_C)$, eliminating penetration while preserving $C^1$ smoothness.
   - **Paradigm 3 (QVI Proximal Fixed-Point Acceleration)**: Wraps coupled state-dependent constraints (e.g. thermoforming QVI) in an outer proximal loop, regularizing the fixed-point map and eliminating slow contraction bottlenecks (13 vs 164 iterations).
   - **Paradigm 4 (Bregman-ADMM Domain Decomposition)**: Splits interface equality via ADMM with Bregman divergences, guaranteeing strict feasibility ($u \in \text{int } K$) within each local solver.
4. **Prime Physical Targets**: Fluid-structure-contact interaction (valves, impact), poroelasticity with cavitation and saturation bounds, and pressurized phase-field fracture with crack irreversibility.
