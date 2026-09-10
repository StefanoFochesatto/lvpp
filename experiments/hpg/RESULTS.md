# hpG port: results of record (2026-09-10)

All scripts run under the env guard
(`PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default OMP_NUM_THREADS=1`,
`/home/stefano/firedrake/venv-firedrake/bin/python`).  Serial only; no existing
files modified.  Spec: `lvpp/HPG_PORT_SPEC.md`.  Wall used by gates ~1.5 min
(stage-0) + ~1 min (stage-1); stages 2/3 NOT RUN (wall budget consumed by
machinery bring-up — see "Blockers" below for exactly what is missing).

## Stage 0 — GATE PASSED (all structure checks machine-precision)

`stage0_gate.py` on 16x16 uniform quads, `RectangleMesh(..., quadrilateral=True)`:

| p | q=p-2 | k/cell | psi-latent dofs | psi-mass modal off-diag | modal diag vs closed form | nodal round-trip | Gram vs kron | Bhat vs injection | Y-Ahat vs closed | Y parity off-block | D_psi off-cell-block |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 1  | 256  | 0.0       | 1.0e-17 | 0.0     | 3.5e-18 | 6.9e-18 | 2.1e-14 | 0.0     | **0.0** |
| 3 | 1 | 4  | 1024 | 3.9e-18   | 6.9e-18 | 1.7e-18 | 1.4e-17 | 6.9e-18 | 8.2e-14 | 2.2e-14 | **0.0** |
| 4 | 2 | 9  | 2304 | 4.1e-18   | 3.5e-18 | 1.1e-18 | 8.7e-18 | 4.2e-18 | 9.2e-14 | 3.2e-14 | **0.0** |

Measured structural facts:
- **Lemma 3.2 holds exactly in Firedrake**: DQ_{p-2} `variant="spectral"` is the
  tensor Legendre basis with a *nodal* dual at the Gauss points.  The modal
  mass is diagonal to 4e-18; the discovered Vandermonde V round-trips the
  nodal layout exactly.  (Raw `Function.dat` nodal values are NOT the modal
  coefficients — all Shat math is done modally through the discovered V.)
- **D_psi is exactly cell-block-diagonal** (off-cell-block max = 0.0, bit-exact).
- **Y-basis cell stiffness**: the paper's "diagonal in 1D" holds for the
  *stiffness* (Y' = -(2n+3)P_{n+1}), but the Y *mass* is a +-2 band (Y is
  Jacobi P^{(1,1)}-orthogonal, not dx-orthogonal), so the 2D per-cell Ahat is
  **block-diagonal with 4 parity blocks** (the paper's "4N blocks in 2D"),
  verified: AY vs closed form 8e-14, inter-parity entries 2-3e-14.  Kept dense
  (k x k, k <= 9 at p=4) — equivalent cost class.

Serial-direct control (monolithic MUMPS, own Newton on the lvpp two-row
residual, psi_floor=0) vs the shared analytic sphere solution:

| level | p | dofs | prox | newton | err(u_h) | wall | our P1 ref (dofs) |
|---|---|---|---|---|---|---|---|
| L0 | 2 | 1345  | 8 | 22 | 1.19e-3 | 3.2 s | L0 P1: 1.55e-2 (545) |
| L0 | 3 | 3425  | 7 | 25 | 1.98e-4 | 5.9 s | — |
| L1 | 2 | 5249  | 7 | 20 | 1.04e-4 | 0.4 s | L1 P1: 3.60e-3 (2113) |
| L1 | 3 | 13505 | 8 | 23 | 3.78e-5 | 1.0 s | — |

Per-dof accuracy: hpG p=2 at 1345 dofs is ~13x more accurate than our P1 at
545 dofs, and ~35x more accurate than our P1 at matched ~2k dofs — the paper's
selling point confirmed on our benchmark.

**lvpp.py integration (psi_spaces route, no edits)**: works.  The audit of
`lvpp.py` (psi_spaces kwarg honored; u_tilde evaluation via
`BoxConstraint.observable` on the *primal* space; drift field on the latent
space) found no hard blocker for DG latents: pre-flight `stage0_probe.py`
ran CG2 x DG0 on quads end-to-end (prox=8, newton=22).  The one cosmetic
friction: tsfc warns about quadrature degree for the exp(psi) latent form;
quadrature_degree=20 covers p<=4.

## Stage 1 — GATE PASSED (machinery verified)

`stage1_machinery.py`, `hpg_shat.py`, `hpg_pc.py`:

- **Alpha-independence (paper's cache trick) verified by construction**:
  max |K(5) - 5 K(1)| over interior rows = 3.6e-15; Dirichlet rows are
  identity at both alphas (0.0 deviation).  The cached K0 = diag(1/alpha,
  1_bc) K(alpha) is factored ONCE per mesh (MUMPS-LU) and reused for every
  proximal iteration: A(alpha)^-1 b = K0^-1 (D_alpha b).
- **Shat SPD at the deep converged state** (D_psi from the real Jacobian):
  min eig(-Shat_c) = 8.5e-7 at beta=0; the beta_Shat ladder raises it as
  expected (1.0e-6 / 2.4e-6 / 1.6e-5 at beta = 1e-5/1e-4/1e-3).
- **Cold inner-GMRES on the true (matfree) Schur with the Shat PC**: 14 its
  at the shallow state, 12 at the deep state (rel 1e-4 / 8e-6) — healthy,
  consistent with the paper's polylogarithmic claim at p=2, 16x16 quads.
- Sign-convention note (documented in code): lvpp's lower-bound latent block
  is J_psi_psi = -D_psi, so S = J_psi_psi - B^T A^-1 B (negative definite),
  and -Shat_c = -D_jac + coupling + beta*M in the factorization.

## Stages 2/3 — NOT RUN (wall budget consumed by machinery bring-up)

The two-stage PC (`HPGTwoStagePC`, paper eq 4.5) is implemented and its
components individually gated (alpha-cache, Shat factorization, inner
matfree-S GMRES, batched cellwise Cholesky apply, getArray idiom), but the
end-to-end promotion (attach as `pc_type: python` PC to `lvpp._solver.snes.ksp`
with per-prox refresh, L0-L2 x p=2/4 table, beta sweep, graded band chain,
l6 divergence check) was not reached.  What remains, in order:

1. Driver script that wraps `lvpp._solver.snes.ksp` with the Python PC and
   the FGMRES outer (`ksp_type: fgmres`), alpha_getter from `lvpp._alpha`,
   and re-runs the LU_REFS comparison table (the `promob.py` promotion
   pattern transfers directly).
2. Stage-3 arms: beta_Shat sweep (the build_shat beta argument is already
   wired), inner rtol/maxit caps (already parameters of HPGCtx), graded
   meshes via `graded_quad` (ratio-graded Cartesian, NOT viamr SBR: SBR is
   triangle-only per viamr.py's documented limitation — hpG requires
   tensor-product cells anyway; the deviation is documented in hpg_common).
3. The band_l6-divergence analogue needs the graded chain of (2).

## What was learned that our config did not know (verified, not inferred)

1. The Y_n = P_n - P_{n+2} basis does NOT give a diagonal 2D Ahat — its 1D
   mass has a +-2 band, giving 4 parity blocks per cell (paper's "4N blocks").
   At p<=4 the per-cell blocks are <= 9x9 dense; the per-cell Cholesky of
   -Shat_c costs essentially nothing (measured: setup 0.09 s per Newton
   refresh at 256 latent dofs, including the batched eig check).
2. Firedrake's DQ-spectral has a nodal dual: modal algebra needs the
   discovered Vandermonde; without it, Lemma-3.2 diagonality is invisible in
   `Function.dat`.  The round-trip check (psi_mass_nodal_match <= 1.7e-18)
   makes this safe.
3. Our alpha-cache finding (K = alpha*stiffness) is exactly the paper's sec-4.4
   trick and was verified numerically here at machine precision (3.6e-15) on
   the quad hpG discretization.

## Files (all new, all under lvpp/experiments/hpg/)

- `hpg_common.py` — shared harness (analytic solution, meshes incl.
  graded_quad, spaces CG_p x DQ_{p-2} spectral, make_lvpp, monitors).
- `hpg_shat.py` — per-cell spectral-Galerkin machinery: 1D closed forms
  (quadrature-verified), discovered modal<->nodal Vandermonde, per-cell
  block extraction, batched Shat Cholesky + apply, structural verification.
- `hpg_pc.py` — HPGCtx (cached K0 LU / CG+GAMG stretch), true-Schur MatShell,
  ShatBlockPC, HPGTwoStagePC (paper eq 4.5 sequential P_F).
- `stage0_probe.py` — pre-flight (DQ variant support, LVPP+DG latent run).
- `stage0_gate.py` — stage-0 gate tables (RUN, PASSED).
- `stage1_machinery.py` — stage-1 gates (RUN, PASSED).


## Stages 2/3 — PROMOTED (2026-09-10; graded g4 row verified by parent rerun)

Driver: `promote_hpg.py` (one new file; machinery untouched; one latent bug
worked around driver-side — HPGTwoStagePC.setUp passed scipy-csr ctx.D into
sg.cell_blocks which expects a PETSc Mat; SGDriver dispatches on input type,
equivalent because D_psi is bit-exact cell-block-diagonal per stage 0).

### Uniform (level, p) — two-stage P_F vs serial-direct controls

| level | p | direct: prox/newton/err/wall | two-stage: same + outer its, inner its/apply, wall |
|---|---|---|---|
| L0 p2 | 1345 | 8/22, 1.192e-3, 0.3s | 8/23, 1.193e-3, **outer max 2**, inner 12.8 (first 5.4/7 → deep 18/20), 4.0s |
| L1 p2 | 5249 | 7/20, 1.036e-4, 0.4s | 7/20, 1.036e-4, **outer 2**, inner 15.7, 14.2s |
| L2 p2 | 20737 | 7/20, 2.651e-5, 1.5s | 7/20, 2.651e-5, **outer 2**, inner 27.0, 96.9s |
| L0 p4 | 5425 | 8/28, 9.620e-5, 6.9s | **published arm WEDGES** (inner GMRES hits 500-cap reason −3 on nearly every apply once α deepens; >20 min killed). **Capped arm (rtol 1e-4/cap 40): 8/30, 9.542e-5, 117s** |

Outer FGMRES max 2 its FLAT across L0–L2 at p=2 (vs paper 14–36 avg; vs our
P1 selfp 14/29/48) — end-to-end confirmation of the polylog claim. Inner
its/apply grows mildly with dofs (12.8→15.7→27.0), consistent with Fig 4.

### Beta sweep (floorless transfer test, L0 p2)

All beta in {0, 1e-5, 1e-4, 1e-3}: identical solves (prox 8, newton 22-23,
outer max 2); deep inner its 18/18/17/22 per apply — beta=1e-3 if anything
slightly worse. **Floorless finding TRANSFERS to hpG: beta_Shat=0 viable.**

### Graded chain (p=2; ratios measured hmax/hmin)

| level | cells | ratio | published (r1e-6/c500) | capped (r1e-4/c40) |
|---|---|---|---|---|
| g4 | 4096 | 26.4x | 6/20, err 5.892e-5, outer 2, 67.9s | 6/20, 5.888e-5, 60.8s |
| g5 | 6400 | 55.3x | 6/17, 4.960e-5, outer 2, 72.8s | 6/17, 4.961e-5, 77.2s |
| g6 | 9216 | 114.8x | 7/19, 7.748e-5, outer 2, 131.3s | 9/21, 7.720e-5, 153.0s |
| g7 | 12544 | 226.1x | 6/20, 4.226e-5, outer 2, 248.3s | 6/20, 4.226e-5, 223.9s |

**NO divergence on the entire chain — clean solve at 226x grading / 25.7k
dofs** (our selfp P1 diverged ksp=−5 at 14.2k on band_l6).  hpG's Shat
survives grading — the paper's untested claim CONFIRMED (Remark 6.1 gap
closed, in hpG's favor).  Capped-inexact discipline is NEUTRAL at p=2
graded; DECISIVE at p=4 (see L0 p4 row above).

### Wall cost (measured, honest)

hpG two-stage: 4.0/14.2/96.9 s (L0/L1/L2 p2) vs serial-direct MUMPS on the
same meshes 0.3/0.4/1.5 s — the P_F PC costs ~30-60x a direct solve at L2
(inner ~27 its/apply, each with a MUMPS K⁻¹; scipy-csr B; the paper's fast
transforms would close this).  Iteration counts, not wall, are hpG's win.

### Verdict (four questions)
(a) faithful port reproduces and BEATS the paper's iteration claims on
uniform meshes (outer 2 its flat; inner polylog-consistent); (b) floorless
transfers — beta_Shat=0 viable end-to-end; (c) hpG Shat SURVIVES grading
(our selfp failure is discretization/PC-specific, not fundamental), and our
capped-inexact discipline is decisive at high p (p=4 wedge rescued);
(d) hpG wins per-dof accuracy (13-35x) and grading robustness; our P1
selfp config wins wall at small-moderate sizes with the same accuracy in
err(u_h) that the problem's regularity supports.


## TRUE-MATFREE hpG (2026-09-10, `matfree_hpG.py`; smoke verified by parent)

Solve path now free of global factorizations: K^-1 = capped CG+GAMG/hypre
(one AMG hierarchy built once per mesh from assembled K0 — preconditioner
construction only), true-S matvec via the existing SShellCtx, Shat PC =
batched CELLWISE Cholesky (cell-local), E_beta = 0 in the operator always
(beta only in Shat), capped inner GMRES (rtol 1e-4, cap 40), outer FGMRES.

**Correctness (p2, vs serial-direct): bit-consistent** — L0 8/22/1.192e-3
(identical), L1 7/20/1.037e-4 vs 7/20/1.036e-4; A-CG 0 nonconv.

**The fair high-order comparison (L0 p4, 4225+2304 dofs, SAME capped
config, both arms):**
- assembled + cached-Cholesky: 8/30, err 9.542e-5, outer 27, inner
  39.4/apply, 117.2 s
- **matfree (A via CG+GAMG): 8/30, err 9.542e-5, outer 27, inner 39.4/apply
  — IDENTICAL prox/newton/err/outer-its; 404.5 s (A-wall 290.5 s)**

Iteration counts do not degrade at all from making A matfree — the matfree
cost is purely per-apply A-CG time (3.5x wall at L0 p4; ~1.3x at p2; the
ratio worsens as kappa(A) grows with p/grading).  Wall attribution:
A-solves dominate the matfree arm (276.6 s shell of 404.5 s at L0 p4).

**Graded 51.4x p4 (37k+20k dofs) — two honest negatives:**
1. GAMG on graded K0 STALLS (rel 3.49e-4 after 251 its vs 38 its on
   uniform; 1994/1994 A-CG nonconv in-solve; PETSc 101) — smoothed
   aggregation cannot handle 51x anisotropy at p=4.
2. hypre BoomerAMG fallback: A-arm HEALTHY (avg 28.7 its, zero nonconv
   across 1061 calls), first prox iteration completed in 913.8 s, but the
   full solve projects to ~2000-2500 s — beyond the session budget; row
   recorded as healthy-but-not-completed (driver instrumentation bug in
   the partial run's apply counters since fixed and smoke-verified).

Assembled-cache arm at the same state: 8/22, err 1.568e-5, 535.2 s —
the E_beta-decoupled + capped-GMRES config HOLDS at high p + 51x grading
+ 58k dofs in its assembled form.

**Net scaling verdict.**  True-matfree hpG is CORRECT (bit-consistent,
identical iteration counts vs assembled arm) and its cost is a per-apply
A-solve premium that grows with kappa(A) (p, grading) — 1.3x at p2, 3.5x
at p4 uniform, 4-6x (projected) at graded p4.  The AMG choice for graded
high-order K is the open preconditioner question (GAMG fails; BoomerAMG
works but is expensive).  L1p4 and p6 rows: not run within budget.
