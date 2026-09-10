"""The saddle-point preconditioner seam.

A :class:`SaddlePreconditioner` is everything :class:`lvpp.solver.LVPP` needs
to know about how the mixed Newton system is (approximately) factored.  The
solver owns the proximal loop, the mixed weak form, and the SNES; the
preconditioner owns the PETSc options that factorize it and, if it needs more
than options can express (a Python PC, a per-iterate cellwise factorization),
it is handed a :class:`SaddleView` to wire itself up.

Why a view and not the solver
-----------------------------
The pre-rewrite hpG port reached into ``lvpp._solver.snes.ksp``,
``lvpp._alpha``, ``lvpp._bcs``, ``lvpp._z``, ``lvpp._J`` and ``lvpp._spaces``
from 13 experiment scripts.  Those accesses were the de-facto extension API.
:class:`SaddleView` makes that API explicit and read-only, so a preconditioner
never needs a reference to the solver object.

Live objects, not copies
------------------------
``alpha`` is a :class:`~firedrake.Constant` and ``z``/``drift`` are
:class:`~firedrake.Function` objects that the solver mutates in place.  A
preconditioner may therefore cache the *reference* at ``install`` time and read
current values in its own ``setUp`` (PETSc calls it once per linear solve, i.e.
per Newton linearization).  This replaces the old ``alpha_getter`` callback.

The Jp-only rule
----------------
``jacobian`` is the **true** Jacobian ``J``; regularization/floors belong on
the preconditioner Jacobian only, because the outer proximal point iteration
needs exact Newton directions (measured: an operator-level floor fails at every
mesh level, ``RESEARCH.md`` Finding 1).  A preconditioner that wants an extra
term returns it from :meth:`SaddlePreconditioner.jacobian_correction` and the
solver adds it to ``Jp`` alone.
"""

import typing
from dataclasses import dataclass

__all__ = [
    "SaddleView",
    "SaddlePreconditioner",
    "JacobianCorrection",
]


@dataclass(frozen=True)
class SaddleView:
    """Read-only handoff from :class:`~lvpp.solver.LVPP` to a preconditioner.

    All object-valued fields are live: the solver mutates them in place, so a
    cached reference stays current.  ``primal`` dofs are contiguous and come
    first in the mixed space, then the latent blocks in ``constraint_unknowns``
    order; ``n_primal`` is the offset (the hpG two-stage PC relies on it).
    """

    z: typing.Any
    """The mixed unknown ``z = (u..., psi...)`` (live ``Function``)."""

    alpha: typing.Any
    """The proximal parameter (live ``Constant``)."""

    drift: tuple
    """Per-latent-block multiplier ``(psi_prev - psi) / alpha`` (live Functions)."""

    constraint_unknowns: tuple
    """For each latent block, the index of the unknown it constrains."""

    primal_spaces: tuple
    """One ``FunctionSpace`` per user unknown."""

    latent_spaces: tuple
    """One ``FunctionSpace`` per latent block."""

    bcs: tuple
    """Essential ``DirichletBC`` objects lifted onto the mixed space."""

    n_primal: int
    """Number of primal dofs (the latent blocks start at this index)."""

    mesh: typing.Any
    """The ``Mesh``."""

    jacobian: typing.Any
    """The true Jacobian as a UFL 2-form (never floored/regularized)."""

    def constraint_measure(self, unknown: int):
        """UFL measure on the mesh of unknown ``unknown`` (``dx`` with the
        solver's form-compiler parameters)."""
        import ufl
        return ufl.Measure("dx", domain=self.primal_spaces[unknown].mesh())


class SaddlePreconditioner(typing.Protocol):
    """How the mixed Newton system is approximately factored.

    Implementations are ordinary objects (not necessarily PETSc types).  The
    default, :class:`lvpp.preconditioners.direct.DirectFactorization`, does
    everything through ``parameters()``; preconditioners that need a Python PC
    implement ``install``.
    """

    def parameters(self) -> dict:
        """PETSc SNES/KSP/PC options for this preconditioner.

        Merged as ``{**DEFAULT_SOLVER_PARAMETERS, **parameters(), **user}``,
        so an explicit user ``solver_parameters`` dict still wins.
        """
        ...

    def jacobian_correction(self, view: SaddleView):
        """Return a 2-form added to ``Jp`` only, or ``None``.

        The form must be built over the mixed trial/test functions of
        ``view.z``'s space.  Default: no correction.
        """
        return None

    def operator_correction(self, view: SaddleView):
        """Legacy diagnostic: return a 2-form added to ``J`` *itself*, or ``None``.

        Only ``DegeneracyFloor(on_operator=True)`` uses this; it violates the
        Jp-only rule and is measured to fail at every mesh level
        (``RESEARCH.md`` Finding 1).  It exists so the rewritten library can
        still reproduce that negative result.  Default: no correction.
        """
        return None

    def install(self, snes, view: SaddleView) -> None:
        """Wire up anything ``parameters()`` cannot express.

        Called once, after the SNES is constructed and before the first solve.
        Default: no-op.
        """

    def finalize(self) -> None:
        """Release any PETSc objects this preconditioner created.

        Called by :meth:`LVPP.close`; safe to call before ``install``.
        """


class JacobianCorrection:
    """Adapter turning the legacy ``jacobian_regularization=`` callable into a
    preconditioner-side correction.

    The original hook had signature ``f(z, z_test, z_trial) -> 2-form``; the
    rewrite keeps that user-facing callable and wraps it here.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, view: SaddleView):
        from firedrake import TestFunction, TrialFunction
        Z = view.z.function_space()
        return self.fn(view.z, TestFunction(Z), TrialFunction(Z))


class PreconditionerBase:
    """No-op defaults for everything except :meth:`parameters`.

    Concrete preconditioners subclass this; they only override what they use.
    """

    def parameters(self) -> dict:
        return {}

    def jacobian_correction(self, view: SaddleView):
        return None

    def operator_correction(self, view: SaddleView):
        return None

    def install(self, snes, view: SaddleView) -> None:
        return None

    def finalize(self) -> None:
        return None

