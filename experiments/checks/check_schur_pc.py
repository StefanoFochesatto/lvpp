"""Schur-fieldsplit runs of the sphere benchmark.

Solves the obstacle problem with the pointwise lower bound psi(r) on the three
uniform refinements of the crossed square mesh and compares the quantities
produced by :class:`lvpp.preconditioners.SchurFieldsplit` -- proximal
iterations, Newton iterations, err(u_h) and the largest outer-GMRES iteration
count -- with the floorless numbers of Finding 6 ("RESOLVED") in
``RESEARCH.md``, which ``lvpp.benchmarks`` stores as ``LU_REFS_P1`` (dofs,
proximal, Newton, error) and ``SCHUR_OUTER_REFS`` (outer Krylov iterations).
Per level, (prox, newton, err(u_h), max outer-GMRES its) = L0 (8, 21, 1.553e-2,
14), L1 (11, 23, 3.599e-3, 29), L2 (8, 19, 8.375e-4, 48).

The option dict is the one owned by
:class:`lvpp.preconditioners.SchurFieldsplit`, selected here with
``preconditioner="schur"`` and no degeneracy floor (``psi_floor=0.0``).

The reported errors carry four significant figures, so they are compared
relatively at 5e-4; the counts must match exactly.

Run from the repository root:

    PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
    OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
    experiments/checks/check_schur_pc.py

Prints one line per level, then ``SCHUR PC OK``; any mismatch exits non-zero.
"""

import sys

from firedrake import (DirichletBC, Function, FunctionSpace, MeshHierarchy,
                       RectangleMesh, SpatialCoordinate, dx, errornorm, grad,
                       inner, sqrt)

from lvpp import LVPP
from lvpp.benchmarks import LU_REFS_P1, SCHUR_OUTER_REFS, psiUFL, uexactUFL


class SolveMonitor:
    """Records the outer-Krylov iteration count of every linear solve.

    PETSc calls a KSP monitor once per iteration; ``its`` restarts at 1 for
    each solve, so groups are split on ``its == 1`` and the max within a group
    is that solve's count.
    """

    def __init__(self):
        self.groups = []

    def __call__(self, ksp, its, rnorm):
        if its == 1:
            self.groups.append(1)
        elif its > 1 and self.groups:
            self.groups[-1] = its


def run(level, mesh):
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    lb = Function(V, name="lb").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    energy = 0.5 * inner(grad(u), grad(u)) * dx

    lv = LVPP(energy=energy, u=u, bounds=(lb, None), bcs=bc,
              alpha_schedule="double_exponential",
              alpha_parameters={"alpha_max": 10.0},
              increment_norm="H1", verbose=False,
              preconditioner="schur",
              form_compiler_parameters={"quadrature_degree": 6},
              name=f"schurL{level}")
    mon = SolveMonitor()
    lv.install_monitor(mon)
    lv.solve(tol=1e-4, max_proximal_iterations=500)
    err = float(errornorm(uexactUFL(r), lv.u_out[0]))
    return lv, mon, err


def main():
    base = RectangleMesh(16, 16, 2.0, 2.0, originX=-2.0, originY=-2.0,
                         diagonal="crossed")
    hierarchy = MeshHierarchy(base, 2)

    print(f"{'lvl':>3} {'dofs':>6} {'prox':>5} {'newton':>7} {'err(u_h)':>10} "
          f"{'maxits':>7}  recorded")
    failures = []
    for level, mesh in enumerate(hierarchy):
        lv, mon, err = run(level, mesh)
        dofs = lv.primal_spaces[0].dim()
        prox = int(lv.proximal_iterations)
        newton = int(sum(lv.newton_iterations))
        maxits = max(mon.groups) if mon.groups else 0
        r_prox, r_newton, r_err, r_dofs = LU_REFS_P1[level]
        r_maxits = SCHUR_OUTER_REFS[level]
        print(f"{level:>3} {dofs:>6} {prox:>5} {newton:>7} {err:>10.3e} "
              f"{maxits:>7}  ({r_dofs}, {r_prox}, {r_newton}, {r_err:.3e}, {r_maxits})")
        if dofs != r_dofs:
            failures.append(f"L{level}: dofs {dofs} != {r_dofs}")
        if prox != r_prox:
            failures.append(f"L{level}: prox {prox} != {r_prox}")
        if newton != r_newton:
            failures.append(f"L{level}: newton {newton} != {r_newton}")
        if abs(err - r_err) > 5e-4 * abs(r_err):
            # The recorded errors are quoted to 4 significant figures, so a
            # relative agreement of ~3e-4 is exact agreement.
            failures.append(f"L{level}: err {err:.6e} != {r_err:.6e}")
        if maxits != r_maxits:
            failures.append(f"L{level}: max outer its {maxits} != {r_maxits}")

    if failures:
        print("\nSCHUR PC MISMATCH:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nSCHUR PC OK (prox/newton/err/maxits all match the recorded table)")


if __name__ == "__main__":
    main()
