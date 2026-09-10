"""Shared analytic benchmark data: the sphere obstacle problem.

One obstacle ``psi`` (a spherical cap glued to a linear far-field) and the
closed-form solution ``uexact`` with free boundary at ``AFREE``.  Lifted out
of ``examples/sphere_lvpp.py``, ``experiments/hpg/hpg_common.py`` and the
archived experiment scripts so the three copies cannot drift; see
``LVPP_REWRITE_SPEC.md`` §12 Q4.

Domain [-2, 2]^2, f = 0, homogeneous data reduction: LVPP solves

    minimize  int 0.5 |grad u|^2 dx   s.t.   u >= psi,   u = uexact on dOmega.

The proximal iteration count on this problem is mesh independent (the LVPP
paper's headline property).
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
    """Obstacle as UFL, from a UFL expression for the radius ``r``."""
    psi0 = np.sqrt(1.0 - r0 * r0)
    dpsi0 = -r0 / psi0
    return conditional(le(r, r0), sqrt(1.0 - r * r), psi0 + dpsi0 * (r - r0))


def uexactUFL(r):
    """Exact solution (and boundary data) as UFL."""
    return conditional(le(r, AFREE), psiUFL(r), -A_ * ln(r) + B_)


LU_REFS_P1 = {
    0: (8, 21, 1.553e-2, 545),
    1: (11, 23, 3.599e-3, 2113),
    2: (8, 19, 8.375e-4, 8321),
}
"""Recorded P1 floorless direct-LU reference: (prox, newton, err(u_h), u-dofs)
per level, on the ``examples/sphere_lvpp.py`` mesh hierarchy."""

SCHUR_OUTER_REFS = {0: 14, 1: 29, 2: 48}
"""Recorded max outer-GMRES iterations per linear solve, floorless Schur
configuration (``RESEARCH.md`` Finding 6, "RESOLVED")."""
