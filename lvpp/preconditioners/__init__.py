"""Preconditioners for the mixed saddle system -- the solver's PC seam.

``LVPP(preconditioner=...)`` accepts four things, all normalized by
:func:`resolve_preconditioner`:

===============  ==================================================
input            result
===============  ==================================================
``None``         :class:`DirectFactorization` (the default)
``"schur"``      a registered name (see :data:`PRECONDITIONERS`)
a ``dict``       :class:`RawOptions` -- legacy "just solver options"
an instance      returned as-is, if it has a callable ``parameters``
===============  ==================================================

A preconditioner owns the PETSc options that factorize the mixed Newton
system and, when options cannot express it, a :class:`SaddleView` to wire
itself up.  See :mod:`lvpp.preconditioners.base` for the seam itself, and
``LVPP_REWRITE_SPEC.md`` §4.3 for why it exists.

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

    ``None`` -> the LU default; a string -> the registered class; a dict ->
    :class:`RawOptions`; an object with a callable ``parameters`` -> itself.
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
