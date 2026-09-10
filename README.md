# lvpp — latent variable proximal point for variational inequalities

`lvpp` solves finite-element variational problems with pointwise inequality
constraints (obstacles, box bounds, gradient bounds, simplex constraints) on
Firedrake, using the latent variable proximal point (LVPP) algorithm of

    J. S. Dokken, P. E. Farrell, B. Keith, I. P. A. Papadopoulos, and
    T. M. Surowiec, "The latent variable proximal point algorithm for
    variational problems with inequality constraints", arXiv:2503.05672 (2025).

Each constrained unknown gets a latent variable psi; a mixed Newton system is
solved at each outer (proximal) iteration, and the bound-preserving output
`u_tilde = grad R*(psi)` is feasible pointwise by construction for any alpha.
The number of proximal iterations is mesh-independent.

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

Constraints are given per unknown as `(lower, upper)` pairs (either side may
be `None`), or as any `Constraint` instance; multiple unknowns are supported
and only constrained ones carry latent blocks.  Pass `energy=` (a UFL 0-form)
or `residual=` (weak 1-form(s)); see the `LVPP` docstring and
`examples/sphere_lvpp.py` for the full interface, alpha schedules, and
diagnostics (`feasibility()`, `complementarity()`, `dual_feasibility()`).
