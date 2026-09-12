"""The ``psi_floor*`` keyword arguments of :class:`lvpp.LVPP`.

Both checks reproduce what the older scripts in ``experiments/archive_2026-09/``
record about the degeneracy floor:

(a) ``psi_floor=1e-2`` together with the Schur-fieldsplit option dict (the
    ``solver_parameters=SP_WINNER, psi_floor=1e-2`` pattern of those scripts)
    still solves, and in the recorded 8 proximal iterations.
(b) ``psi_floor_operator=True`` must FAIL: putting the floor on the operator
    itself violates the Jp-only rule and diverges at every level
    (``RESEARCH.md`` Finding 1).  This is a negative control for the
    ``operator_correction`` plumbing: if it stops failing, the operator
    correction is not reaching ``J``.

Run with the env guard from the repository root:

    PETSC_DIR=/home/stefano/firedrake/petsc PETSC_ARCH=arch-firedrake-default \
    OMP_NUM_THREADS=1 /home/stefano/firedrake/venv-firedrake/bin/python \
    experiments/checks/check_deprecated_paths.py
"""

import sys

from firedrake import (ConvergenceError, DirichletBC, Function, FunctionSpace,
                       RectangleMesh, SpatialCoordinate, dx, grad, inner,
                       sqrt)

from lvpp import LVPP, LVPPConvergenceError, SchurFieldsplit
from lvpp.benchmarks import psiUFL, uexactUFL

FCP = {"quadrature_degree": 6}


def make(level=0, **overrides):
    mesh = RectangleMesh(16 * 2 ** level, 16 * 2 ** level, 2.0, 2.0,
                         originX=-2.0, originY=-2.0, diagonal="crossed")
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V, name="u")
    x, y = SpatialCoordinate(mesh)
    r = sqrt(x * x + y * y)
    lb = Function(V, name="lb").interpolate(psiUFL(r))
    bc = DirichletBC(V, uexactUFL(r), "on_boundary")
    kw = dict(energy=0.5 * inner(grad(u), grad(u)) * dx, u=u,
              bounds=(lb, None), bcs=bc,
              alpha_schedule="double_exponential",
              alpha_parameters={"alpha_max": 10.0},
              increment_norm="H1", verbose=False,
              form_compiler_parameters=FCP, name="deprecated")
    kw.update(overrides)
    return LVPP(**kw), V


def main():
    failures = []

    # (a) floored Schur config still solves (archive pattern: solver_parameters
    #     carries the PC, psi_floor carries the floor).
    lv, _ = make(psi_floor=1e-2, solver_parameters=SchurFieldsplit().parameters())
    try:
        lv.solve(tol=1e-4, max_proximal_iterations=500)
        print(f"(a) psi_floor=1e-2 + schur: prox={lv.proximal_iterations} "
              f"newton={sum(lv.newton_iterations)} alpha={lv.alpha:.3e} -> converged")
        if lv.proximal_iterations != 8:
            failures.append(f"(a) prox {lv.proximal_iterations} != 8")
    except Exception as exc:  # noqa: BLE001
        print(f"(a) FAILED: {type(exc).__name__}: {exc}")
        failures.append("(a) floored Schur config did not converge")

    # (b) operator floor must fail (negative control).
    lv2, _ = make(psi_floor=1e-2, psi_floor_operator=True)
    try:
        lv2.solve(tol=1e-4, max_proximal_iterations=10)
        print(f"(b) psi_floor_operator=True CONVERGED (prox="
              f"{lv2.proximal_iterations}) -- the recorded divergence did NOT "
              f"reproduce; operator correction may not reach J")
        failures.append("(b) operator floor unexpectedly converged")
    except (LVPPConvergenceError, ConvergenceError) as exc:
        print(f"(b) psi_floor_operator=True failed as recorded: "
              f"{type(exc).__name__}")
    except Exception as exc:  # noqa: BLE001
        print(f"(b) psi_floor_operator=True raised {type(exc).__name__}: {exc} "
              f"(treated as the recorded failure)")
        failures.append(f"(b) unexpected exception type {type(exc).__name__}")

    if failures:
        print("\nDEPRECATED PATHS MISMATCH:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nDEPRECATED PATHS OK")


if __name__ == "__main__":
    main()
