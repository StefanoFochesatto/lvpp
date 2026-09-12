"""The saddle-point preconditioner interface.

A :class:`SaddlePreconditioner` is everything :class:`lvpp.solver.LVPP` needs to
know about how the mixed Newton system is (approximately) factored.  The solver
owns the proximal loop, the mixed weak form and the SNES; the preconditioner
owns the PETSc options that factorize the mixed system and, when options cannot
express what it wants (a Python PC, a per-iterate cellwise factorization), it
is handed a :class:`SaddleView` to wire itself up.

The view, not the solver
------------------------
A preconditioner never needs a reference to the solver object.  The live mixed
unknown, the proximal parameter, the multiplier fields, the spaces and the true
Jacobian all arrive in a :class:`SaddleView`, which is read-only by
construction; nothing else about the solver is exposed.  The older scripts in
``experiments/archive_2026-09/`` reach for the same quantities through private
attributes of the solver (``_solver.snes.ksp``, ``_alpha``, ``_bcs``, ``_z``,
``_J`` and ``_spaces``), and the view makes that access explicit and read-only
instead.

Live objects, not copies
------------------------
``alpha`` is a :class:`~firedrake.Constant` and ``z``/``drift`` are
:class:`~firedrake.Function` objects that the solver mutates in place, so a
preconditioner may cache the reference at :meth:`install` time and read current
values in its own ``setUp``; PETSc calls that once per linear solve, hence once
per Newton linearization.

The Jp-only rule
----------------
``jacobian`` is the **true** Jacobian ``J``.  A floor or regularization belongs
on the preconditioner Jacobian ``Jp`` alone, never on ``J``: the outer proximal
point iteration is a contractive fixed point map that needs exact Newton
directions, and an operator-level floor perturbs those directions and is
measured to fail at every mesh level (``RESEARCH.md`` Finding 1).  A
preconditioner that wants an extra term returns it from
:meth:`SaddlePreconditioner.jacobian_correction` and the solver adds it to
``Jp`` alone.
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

    A view is what a preconditioner gets in place of the solver object.  It
    carries the mixed unknown ``z``, the proximal parameter ``alpha``, the
    multiplier fields, the spaces the blocks live on, the essential boundary
    conditions and the true Jacobian.  All object-valued fields are live: the
    solver mutates them in place, so a cached reference stays current and a
    preconditioner may read the values afresh at each assembly.

    The primal dofs are contiguous and come first in the mixed space, followed
    by the latent blocks in ``constraint_unknowns`` order; ``n_primal`` is the
    offset at which the latent blocks begin.  A preconditioner can therefore
    index into ``split(z)`` without asking the solver for the block structure,
    which is what the two-stage preconditioner hpG (4.5) does.
    """

    z: typing.Any
    """The mixed unknown ``z = (u..., psi...)`` (a live ``Function``)."""

    alpha: typing.Any
    """The proximal parameter (a live ``Constant``)."""

    drift: tuple
    """Per latent block, the multiplier ``(psi_prev - psi) / alpha`` (live Functions)."""

    constraint_unknowns: tuple
    """For each latent block, the index of the user unknown it constrains."""

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
    """The true Jacobian as a UFL 2-form (never floored or regularized)."""

    def constraint_measure(self, unknown: int):
        """UFL measure ``dx`` on the mesh of the user unknown ``unknown``.

        A correction term for a latent block is integrated over the region that
        block constrains, whose mesh is the mesh of the primal unknown."""
        import ufl
        return ufl.Measure("dx", domain=self.primal_spaces[unknown].mesh())


class SaddlePreconditioner(typing.Protocol):
    """How the mixed Newton system is approximately factored.

    Implementations are ordinary objects, not necessarily PETSc types.  Only
    :meth:`parameters` is needed in practice: the default,
    :class:`lvpp.preconditioners.direct.DirectFactorization`, does everything
    through it.  A preconditioner that must touch PETSc objects, or assemble
    something per iterate, also implements :meth:`install`, and one that wants
    an extra term on the preconditioner Jacobian implements
    :meth:`jacobian_correction`.
    """

    def parameters(self) -> dict:
        """PETSc SNES/KSP/PC options for this preconditioner.

        They are merged as ``{**DEFAULT_SOLVER_PARAMETERS, **parameters(),
        **user}``, so an explicit user ``solver_parameters`` dict still wins.
        """
        ...

    def jacobian_correction(self, view: SaddleView):
        """Return a 2-form to be added to ``Jp`` alone, or ``None``.

        The term lands on the preconditioner Jacobian only, as the Jp-only rule
        in the module docstring requires.  The form must be built over the
        mixed trial and test functions of ``view.z``'s space.  Default: no
        correction.
        """
        return None

    def operator_correction(self, view: SaddleView):
        """Return a 2-form to be added to ``J`` *itself*, or ``None``.

        This exists to demonstrate why the Jp-only rule holds, not to be used.
        Only ``DegeneracyFloor(on_operator=True)`` returns a form here, for
        which the solver sets ``Jp = J = J_eff`` (and, in that case, discards
        any ``jacobian_regularization``).  An operator-level floor is measured
        to fail at every mesh level (``RESEARCH.md`` Finding 1), because the
        outer proximal point iteration then takes inexact Newton directions, so
        this diagnostic is off by default.  Default: no correction.
        """
        return None

    def install(self, snes, view: SaddleView) -> None:
        """Wire up anything ``parameters()`` cannot express.

        Called once, after the SNES is constructed and before the first solve;
        this is where a preconditioner caches the live view references it needs
        in its own ``setUp``.  Default: no-op.
        """

    def finalize(self) -> None:
        """Release any PETSc objects this preconditioner created.

        Called by :meth:`LVPP.close`, and safe to call before ``install``.
        """


class JacobianCorrection:
    """Adapter turning the ``jacobian_regularization=`` callable accepted by
    :class:`~lvpp.solver.LVPP` into a preconditioner-side correction.

    The callable has signature ``f(z, z_test, z_trial) -> 2-form``, a UFL form
    over the mixed space written by the user; this wrapper supplies the test and
    trial functions of ``view.z``'s space and hands the result back for ``Jp``.
    """

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, view: SaddleView):
        from firedrake import TestFunction, TrialFunction
        Z = view.z.function_space()
        return self.fn(view.z, TestFunction(Z), TrialFunction(Z))


class PreconditionerBase:
    """No-op defaults for every method except :meth:`parameters`.

    Concrete preconditioners subclass this and override only what they use, so
    one that needs no correction, no PETSc wiring and nothing to release is a
    single ``parameters()`` method.
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