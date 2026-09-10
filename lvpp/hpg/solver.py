"""``HPG``: the hpG preset -- a thin wrapper around :class:`LVPP`.

hpG is the *hierarchical Proximal-Galerkin* framework of Papadopoulos
(arXiv:2412.13733): the same latent-variable proximal point loop as LVPP, on
a ``CG_p x DQ_{p-2}`` tensor-product discretization, preconditioned by the
cellwise spectral-Galerkin two-stage ``P_F``.  Formally it is **two orthogonal
axes**:

===============  ==========================================  ==========================
axis             choice                                      module
===============  ==========================================  ==========================
discretization   ``CG_p`` primal x ``DQ_{p-2}`` spectral      :mod:`lvpp.hpg.spaces`
preconditioner   sequential ``P_F`` (eq. 4.5)                :mod:`lvpp.hpg.twostage`
===============  ==========================================  ==========================

This class is a *preset builder*: it picks both axes and hands them to
:class:`~lvpp.solver.LVPP`.  It overrides **no algorithm step** -- not the
proximal loop, not the schedule, not the stopping rule, not the Newton
solver.  That is deliberate and is the executable test of the rewrite's
decomposition (``LVPP_REWRITE_SPEC.md`` §0/§5.4): if it ever needed to override
something beyond ``__init__``, an axis would have been mis-modeled, and the
right fix would be a genuinely different algorithm subclass -- not edits
here.

The two axes compose: any LVPP user can pass
``preconditioner=HPGTwoStage()`` to a plain LVPP (the preconditioner), or
reach the discretization alone through :class:`~lvpp.hpg.spaces.HPGDiscretization`
and drive it with their own preconditioner.
"""

from lvpp.hpg.spaces import HPGDiscretization
from lvpp.hpg.twostage import HPGTwoStage
from lvpp.solver import LVPP

__all__ = ["HPG"]


def _as_discretization(discretization):
    """``HPGDiscretization`` as-is, or ``(mesh, p)`` as the convenience pair.

    The pair uses :meth:`HPGDiscretization._on`, the internal builder for an
    already-built tensor-product mesh (``uniform``/``graded`` build their own
    meshes, so neither applies to a caller's mesh).
    """
    if isinstance(discretization, HPGDiscretization):
        return discretization
    if (isinstance(discretization, tuple) and len(discretization) == 2
            and isinstance(discretization[1], int)):
        mesh, p = discretization
        return HPGDiscretization._on(mesh, p)
    raise TypeError(
        "discretization must be an HPGDiscretization or a (mesh, p) tuple; "
        f"got {discretization!r}")


class HPG(LVPP):
    """hpG preset: ``CG_p x DQ_{p-2}`` spectral with the two-stage ``P_F``.

    Parameters
    ----------
    discretization : HPGDiscretization or (mesh, p)
        The hpG spaces.  A ``(mesh, p)`` tuple is accepted for convenience
        and builds ``HPGDiscretization`` on that tensor-product mesh.
    u : Function
        The primal unknown, living on ``discretization.primal`` (``CG_p``).
    bounds : Constraint or (lower, upper)
        As :class:`~lvpp.solver.LVPP`; the single obstacle ``(lb, None)`` in
        the benchmark.
    bcs : DirichletBC or list, optional
        Essential BCs.
    preconditioner : SaddlePreconditioner, optional
        Defaults to :class:`~lvpp.hpg.twostage.HPGTwoStage` -- that is the
        hpG axis.  Pass another preconditioner (e.g. a direct LU) to keep the
        discretization and change the factorization.
    **lvpp_kwargs
        Forwarded verbatim to :class:`~lvpp.solver.LVPP`; every schedule and
        stopping default is LVPP's.  The recorded hpG numbers use
        ``alpha_schedule="double_exponential"``,
        ``alpha_parameters={"alpha_max": 10.0}``, ``increment_norm="H1"`` and
        ``solve(tol=1e-4)``.

    Attributes
    ----------
    discretization : HPGDiscretization
        The chosen spaces (``disc.primal`` / ``disc.latent``), so callers and
        tests can reach them without rebuilding.
    """

    def __init__(self, discretization, u, bounds, *, bcs=None,
                 preconditioner=None, **lvpp_kwargs):
        disc = _as_discretization(discretization)
        if preconditioner is None:
            preconditioner = HPGTwoStage()
        super().__init__(u=u, bounds=bounds, bcs=bcs,
                         latent_spaces=[disc.latent],
                         preconditioner=preconditioner, **lvpp_kwargs)
        self.discretization = disc
