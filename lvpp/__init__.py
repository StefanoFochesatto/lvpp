"""The lvpp package implements the latent variable proximal point (LVPP)
algorithm for variational problems with pointwise inequality constraints,
following Dokken, Farrell, Keith, Papadopoulos, & Surowiec (2025).

``LVPP`` in :mod:`lvpp.solver` is the solver itself: it owns the proximal loop
over the mixed system (2.7a)-(2.7b), the alpha schedule, and the stopping
rule.  The constraint families live in :mod:`lvpp.constraints` (box bounds) and
:mod:`lvpp.legendre` (the entropy-based functions whose gradients give the
feasible reconstructions); :mod:`lvpp.preconditioners` collects the ways the
mixed Newton systems are approximately factored; :mod:`lvpp.schedules` holds
the step-size rules and the stopping rules; and :mod:`lvpp.hpg` is the
hierarchical Proximal-Galerkin preset built on top of the solver.

The constraint families, the schedules, and the assembled-system helpers are
importable without Firedrake's solver machinery; only :class:`LVPP` itself
needs it, since that is where the SNES and the proximal loop are built.  Import
:mod:`lvpp.hpg` separately when the preset is wanted.
"""

from .constraints import BoxConstraint, Constraint
from .legendre import (FermiDirac, GibbsSimplex, Hellinger, LegendreFunction,
                       ShannonLower, ShannonUpper)
from .preconditioners import (DegeneracyFloor, DirectFactorization,
                              PreconditionerBase, RawOptions,
                              SaddlePreconditioner, SaddleView,
                              SchurFieldsplit, resolve_preconditioner)
from .schedules import (AlphaPlateau, AlphaSchedule, PrimalIncrement,
                        StoppingRule, make_schedule)
from .solver import DEFAULT_SOLVER_PARAMETERS, LVPP, LVPPConvergenceError

__all__ = [
    "LVPP",
    "LVPPConvergenceError",
    "DEFAULT_SOLVER_PARAMETERS",
    "BoxConstraint",
    "Constraint",
    "LegendreFunction",
    "ShannonLower",
    "ShannonUpper",
    "FermiDirac",
    "Hellinger",
    "GibbsSimplex",
    "SaddleView",
    "SaddlePreconditioner",
    "PreconditionerBase",
    "DirectFactorization",
    "RawOptions",
    "DegeneracyFloor",
    "SchurFieldsplit",
    "resolve_preconditioner",
    "AlphaSchedule",
    "StoppingRule",
    "PrimalIncrement",
    "AlphaPlateau",
    "make_schedule",
]
