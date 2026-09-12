"""Preconditioners for the mixed saddle system.

``LVPP(preconditioner=...)`` accepts four forms, and
:func:`resolve_preconditioner` normalizes all of them.  ``None`` gives
:class:`DirectFactorization`, the default.  A string names a registered
preconditioner (see :data:`PRECONDITIONERS`).  A dict is taken as raw PETSc
options and wrapped in :class:`RawOptions`.  An object with a callable
``parameters`` is returned unchanged.

A preconditioner owns the PETSc options that factorize the mixed Newton system
and, when options cannot express what it needs, a
:class:`~lvpp.preconditioners.base.SaddleView` through which it wires itself up.
See :mod:`lvpp.preconditioners.base` for the interface itself.

Nothing here imports Firedrake at module load: the UFL/geometry work happens
inside the methods that need it.
"""

from .base import (
    JacobianCorrection,
    PreconditionerBase,
    SaddlePreconditioner,
    SaddleView,
)
from .direct import DirectFactorization, RawOptions
from .floor import DegeneracyFloor
from .schur import SchurFieldsplit

__all__ = [
    "SaddleView",
    "SaddlePreconditioner",
    "PreconditionerBase",
    "JacobianCorrection",
    "DirectFactorization",
    "RawOptions",
    "DegeneracyFloor",
    "SchurFieldsplit",
    "PRECONDITIONERS",
    "resolve_preconditioner",
]

#: The names accepted as a ``preconditioner="..."`` string.
PRECONDITIONERS = {
    "direct": DirectFactorization,
    "lu": DirectFactorization,
    "schur": SchurFieldsplit,
}


def resolve_preconditioner(spec):
    """Normalize a ``preconditioner=`` argument to a preconditioner object.

    ``None`` gives the LU default, a string names a registered class, a dict
    becomes :class:`RawOptions`, and an object with a callable ``parameters``
    is returned unchanged; anything else raises ``TypeError``.
    """
    if spec is None:
        return DirectFactorization()
    if isinstance(spec, str):
        try:
            return PRECONDITIONERS[spec]()
        except KeyError:
            valid = ", ".join(sorted(PRECONDITIONERS))
            raise ValueError(
                f"unknown preconditioner {spec!r}; valid names: {valid}") from None
    if isinstance(spec, dict):
        return RawOptions(spec)
    if callable(getattr(spec, "parameters", None)):
        return spec
    raise TypeError(
        "preconditioner must be None, a name, an options dict, or an object "
        f"with a callable parameters(); got {spec!r}")