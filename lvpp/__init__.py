"""LVPP: latent variable proximal point algorithm for variational inequalities.

Public surface
--------------
``LVPP`` is the solver; :mod:`lvpp.legendre` and :mod:`lvpp.constraints`
define the constraint families; :mod:`lvpp.preconditioners` defines how the
mixed Newton systems are factored; :mod:`lvpp.schedules` defines the proximal
step-size and stopping rules; :mod:`lvpp.hpg` is the hierarchical
Proximal-Galerkin preset.

Everything here is importable without Firedrake's solver machinery except
:class:`LVPP` itself.
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
