"""Options-only preconditioners: the default LU, and the legacy raw dict.

Both classes do everything through :meth:`parameters`; neither needs a
:class:`~lvpp.preconditioners.base.SaddleView`, and neither imports Firedrake.
"""

from .base import PreconditionerBase

__all__ = ["DirectFactorization", "RawOptions"]


class DirectFactorization(PreconditionerBase):
    """The solver default: one ``preonly`` + LU factorization per Newton step.

    The saddle system is ``2 x 2`` block with a possibly degenerate latent
    block, so a direct factorization is the only preconditioner that is
    unconditionally correct (``RESEARCH.md`` Finding 1: it reproduces the
    reference Newton counts at every level).  It is also the wall-clock
    baseline every iterative configuration is measured against.
    """

    def __init__(self, solver_type="mumps"):
        self.solver_type = solver_type

    def parameters(self):
        return {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": self.solver_type,
        }

    def __repr__(self):
        return f"DirectFactorization(solver_type={self.solver_type!r})"


class RawOptions(PreconditionerBase):
    """A bare PETSc option dict used as a preconditioner.

    ``LVPP(preconditioner={...})`` keeps the pre-rewrite ergonomics of
    ``solver_parameters={...}``: the dict is merged over the defaults exactly
    as given, with no interpretation.  Use this for one-off options and for
    reproducing archived configurations; reach for a named preconditioner
    class when the options describe something with a name.
    """

    def __init__(self, options):
        self.options = dict(options)

    def parameters(self):
        return dict(self.options)

    def __repr__(self):
        return f"RawOptions({self.options!r})"
