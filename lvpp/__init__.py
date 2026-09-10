"""LVPP: latent variable proximal point algorithm for variational inequalities."""

from .constraints import BoxConstraint, Constraint
from .legendre import (FermiDirac, GibbsSimplex, Hellinger, LegendreFunction,
                       ShannonLower, ShannonUpper)
from .lvpp import DEFAULT_SOLVER_PARAMETERS, LVPP, LVPPConvergenceError

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
]
