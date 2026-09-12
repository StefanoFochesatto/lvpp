"""Analytic data for the sphere obstacle problem, shared by the example and
the experiments.

The obstacle ``psi`` is a spherical cap of radius ``r0`` joined continuously to
a linear far field, and ``uexact`` is the closed-form solution of the obstacle
problem whose free boundary sits at radius ``AFREE``: the solution coincides
with the cap over the disk ``r <= AFREE`` and leaves it outside, where it is
the logarithmic ``-A ln r + B``.  On the square ``[-2, 2]^2`` with ``f = 0``
and homogeneous data reduction, LVPP solves

    minimize  int 0.5 |grad u|^2 dx   s.t.   u >= psi,   u = uexact on dOmega.

The proximal iteration count on this problem is mesh independent, which is the
headline property of the LVPP paper, so the example and the experiments use it
as the smallest nontrivial check of the solver.

The dictionaries ``LU_REFS_P1`` and ``SCHUR_OUTER_REFS`` hold measured
reference values from runs of ``examples/sphere_lvpp.py`` on its mesh
hierarchy, kept here so the example and the experiments compare against a
single copy of the numbers instead of each carrying its own.
"""

import numpy as np
from firedrake import conditional, le, ln, sqrt

__all__ = ["r0", "AFREE", "A_", "B_", "psiUFL", "uexactUFL",
           "LU_REFS_P1", "SCHUR_OUTER_REFS"]

r0 = 0.9
"""Cap radius: psi is a spherical cap for r <= r0, linear beyond."""

AFREE = 0.697965148223374
"""Radius of the free boundary (where u leaves the obstacle)."""

A_ = 0.680259411891719
B_ = 0.471519893402112
"""Far-field coefficients: u = -A ln r + B for r > AFREE."""


def psiUFL(r):
    """The obstacle as a UFL expression of the radius ``r``: the sphere
    ``sqrt(1 - r**2)`` on the cap, continued by its tangent line of slope
    ``dpsi0`` so the obstacle is C^1 across ``r = r0``."""
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    """The exact solution as a UFL expression of ``r``: the obstacle on the
    cap and the logarithmic far field beyond ``AFREE``.  This is also the
    Dirichlet data on ``dOmega``."""
    return conditional(le(r, AFREE), psiUFL(r), -A_ * ln(r) + B_)


LU_REFS_P1 = {
    0: (8, 21, 1.553e-2, 545),
    1: (11, 23, 3.599e-3, 2113),
    2: (8, 19, 8.375e-4, 8321),
}
"""Measured reference for the floorless direct-LU configuration: per mesh
level, ``(prox, newton, err(u_h), u-dofs)`` -- the proximal and Newton
iteration counts, the error of the reconstructed ``u_tilde``, and the number of
unknown dofs, on the mesh hierarchy of ``examples/sphere_lvpp.py``."""

SCHUR_OUTER_REFS = {0: 14, 1: 29, 2: 48}
"""Measured maximum outer-GMRES iterations per linear solve, floorless Schur
configuration (``RESEARCH.md`` Finding 6, "RESOLVED")."""
