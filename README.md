# lvpp: latent variable proximal point for variational inequalities

`lvpp` solves finite-element variational problems with pointwise inequality
constraints (obstacles, box bounds, gradient bounds, simplex constraints) on
Firedrake.  The algorithm is the latent variable proximal point (LVPP) method
of

    J. S. Dokken, P. E. Farrell, B. Keith, I. P. A. Papadopoulos, and
    T. M. Surowiec, "The latent variable proximal point algorithm for
    variational problems with inequality constraints", CMAME 445:118181
    (2025); arXiv:2503.05672.

Every constrained unknown carries a latent variable `psi`, and each proximal
iteration solves the mixed Newton system (2.7a)-(2.7b) of that paper.  The
bound-preserving output is the reconstruction `u_tilde = grad R*(psi)`, which
is feasible pointwise by construction for any alpha.  The number of proximal
iterations is independent of the mesh.

`lvpp.hpg` holds the hierarchical proximal Galerkin (hpG) preset of

    I. P. A. Papadopoulos, "Hierarchical proximal Galerkin: a fast hp-FEM
    solver for variational problems with pointwise inequality constraints",
    arXiv:2412.13733 (2026).

It is the same proximal loop and the same saddle system, discretized with
`CG_p x DQ_{p-2}` spectral hierarchical elements on tensor-product cells and
factored with the two-stage `P_F` preconditioner of that paper (eq. 4.5).

## Install

Inside a Firedrake venv:

    pip install -e .

## Quickstart (obstacle problem)

    from firedrake import *
    from lvpp import LVPP

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    x, y = SpatialCoordinate(mesh)
    psi = Function(V).interpolate(0.1 - (x - 0.5)**2 - (y - 0.5)**2)  # obstacle
    f = Constant(1.0)
    lvpp = LVPP(energy=(0.5 * inner(grad(u), grad(u)) - f * u) * dx,
                u=u, bounds=(psi, None))
    lvpp.solve(tol=1e-8)
    print("feasible output min:", lvpp.u_tilde[0].dat.data_ro.min(),
          ">= obstacle min:", psi.dat.data_ro.min())

Constraints are given per unknown as `(lower, upper)` pairs (either side may be
`None`), or as any `Constraint` instance; multiple unknowns are supported and
only constrained ones carry latent blocks.  Pass `energy=` (a UFL 0-form) or
`residual=` (weak 1-form(s)); see the `LVPP` docstring and
`examples/sphere_lvpp.py` for the full interface, alpha schedules, and
diagnostics (`feasibility()`, `complementarity()`, `dual_feasibility()`).

## hpG quickstart (high-order, tensor-product cells)

    from lvpp.hpg import HPG, HPGDiscretization

    disc = HPGDiscretization.uniform(16, p=2)     # 16x16 quads, CG2 x DQ0
    u = Function(disc.primal)
    hp = HPG(disc, u, bounds=(psi, None), bcs=bc,
             alpha_parameters={"alpha_max": 10.0}, increment_norm="H1")
    hp.solve(tol=1e-4)

`HPGDiscretization` also builds geometrically graded meshes
(`HPGDiscretization.graded(n, p, ratio)`) and 1D/3D (`dim=1, 3`); hpG requires
tensor-product cells (interval, quadrilateral, hexahedron).

### The matrix-free A-action

`HPGTwoStage` applies `A(alpha)^-1` either from a cached MUMPS factorization of
the alpha-free `K0` (default, `a_action="lu"`) or, with `a_action="gamg"`, from
a **capped CG + AMG solve on the same `K0`** — the matrix-free option, which
leaves no global factorization on the apply path (the cellwise `Shat` Cholesky
and the preconditioner assembly are local):

    HPG(disc, u, bounds, preconditioner=HPGTwoStage(a_action="gamg"))

Measured by `experiments/checks/check_hpg_matfree.py`:

* **uniform meshes: identical to the cached-factorization path** — L0 p=2 gives
  prox 8, newton 22, `err = 1.191910e-03` for both, with A-CG 14.7 its/call
  (max 16) and zero non-converged solves;
* **graded meshes: the A-solve saturates its cap** — on a 26.4x graded mesh, at
  `a_rtol=1e-8` all 569 A-CG calls hit the 60-iteration cap; loosening to
  `a_rtol=1e-6` leaves 60 of 602 saturated.  The outer FGMRES converges
  (2–3 its) and the result is close to, but not identical with, the
  cached-factorization path (prox 10 vs 8, `err 1.03e-4` vs `9.9e-5`).  AMG on
  a strongly graded stiffness matrix is weak here; `a_rtol` / `a_maxit` are the
  knobs.

## Layout

| module | contents |
|---|---|
| `lvpp/solver.py` | `LVPP`, the proximal loop (schedule, stopping, SNES, diagnostics) |
| `lvpp/assembly.py` | `ProblemSpec` + `MixedSystem`: spaces, residual, Jacobians, diagnostic forms |
| `lvpp/schedules.py` | alpha schedules and stopping rules (`PrimalIncrement`, `AlphaPlateau`) |
| `lvpp/constraints.py` | `Constraint` interface and `BoxConstraint` |
| `lvpp/legendre.py` | Legendre functions (Shannon, Fermi-Dirac, Hellinger, Gibbs simplex) |
| `lvpp/preconditioners/` | the saddle-point preconditioner interface: `DirectFactorization`, `DegeneracyFloor`, `SchurFieldsplit`, `RawOptions` |
| `lvpp/hpg/` | the hpG preset: spaces, per-cell spectral-Galerkin algebra, two-stage preconditioner |
| `lvpp/benchmarks.py` | shared analytic sphere-obstacle data |

## Preconditioners

The mixed Newton system is factored through a small strategy interface
(`lvpp.preconditioners.SaddlePreconditioner`), so the algorithm in
`LVPP.solve` never knows how the linear algebra is done:

    LVPP(..., preconditioner=None)          # direct LU/MUMPS (default)
    LVPP(..., preconditioner="schur")       # floorless Schur fieldsplit
    LVPP(..., preconditioner=HPGTwoStage()) # the hpG two-stage P_F
    LVPP(..., preconditioner={"ksp_type": ...})   # raw PETSc options

Floors and other regularizations live on the preconditioner Jacobian `Jp`
only, never on the operator `J`: the outer proximal point iteration needs
exact Newton directions (`DegeneracyFloor(on_operator=True)` is a documented
diagnostic and is measured to fail).

## Keyword spellings

The preferred keywords are `latent_spaces=`, `alpha_schedule=`, and
`preconditioner=`; the earlier spellings `psi_spaces=`, `alpha_rule=`,
`psi_floor=`, `psi_floor_drift=`, `psi_floor_operator=`, and
`jacobian_regularization=` are also accepted, and raise a
`DeprecationWarning`.  Scripts that read the mixed state directly through the
private attributes (`lvpp._z`, `lvpp._alpha`, `lvpp._solver`, ...) are
unaffected; the public handles (`snes`, `ksp`, `pc`, `matrix()`, `constraints`,
`primal_spaces`, `bcs`, `alpha_constant`, `install_monitor`) are the supported
surface.

See `LVPP_DESIGN.md` for the design, and `experiments/checks/` for the checks
that pin the recorded numbers.
