"""``HPG``: the hpG preset -- discretization plus preconditioner.

hpG is the *hierarchical Proximal-Galerkin* framework of Papadopoulos
(arXiv:2412.13733): the same latent-variable proximal point loop as LVPP, on a
``CG_p x DQ_{p-2}`` tensor-product discretization, preconditioned by the
cellwise spectral-Galerkin two-stage ``P_F`` of section 4.4.  The
discretization and the preconditioner are independent choices:

    discretization   ``CG_p`` primal x ``DQ_{p-2}`` spectral   :mod:`lvpp.hpg.spaces`
    preconditioner   sequential ``P_F`` (eq. 4.5)              :mod:`lvpp.hpg.twostage`

The :class:`HPG` preset makes both choices for a caller and hands them to
:class:`~lvpp.solver.LVPP`, whose every algorithmic step it leaves alone --
the proximal loop, the alpha schedule, the stopping rule and the Newton solver
are LVPP's.  Either choice is also usable on its own: any LVPP accepts
``preconditioner=HPGTwoStage()``, and the discretization is reachable as
:class:`~lvpp.hpg.spaces.HPGDiscretization` for a caller with their own
preconditioner.
"""

from lvpp.hpg.spaces import HPGDiscretization
from lvpp.hpg.twostage import HPGTwoStage
from lvpp.solver import LVPP

__all__ = ["HPG"]


def _as_discretization(discretization):
    """The ``HPGDiscretization`` itself, or ``(mesh, p)`` as the convenience pair.

    A pair is built with :meth:`HPGDiscretization._on`, the constructor for a
    tensor-product mesh the caller already has; :meth:`~HPGDiscretization.uniform`
    and :meth:`~HPGDiscretization.graded` build their own meshes and so do not
    apply here.
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
    """HPG applies the hpG discretization and two-stage preconditioner to the
    LVPP proximal point loop; it is a preset, not a second solver.

    An HPG object owns an :class:`~lvpp.hpg.spaces.HPGDiscretization`: the
    tensor-product ``CG_p`` primal space carrying the obstacle unknown ``u``,
    and the cellwise discontinuous ``DQ_{p-2}`` spectral latent space carrying
    the multiplier ``psi`` (section 4.1 of hpG).  It hands that latent space to
    LVPP and, unless the caller passes another one, installs the sequential
    ``P_F`` preconditioner of hpG section 4.4 as
    :class:`~lvpp.hpg.twostage.HPGTwoStage`.

    The discretization and the preconditioner are independent choices, which
    is the central point about this preset.  The pair ``CG_p x DQ_{p-2}`` is
    the inf-sup stable one (Lemma B.3 of Keith & Surowiec (2024)), and its
    tensor-product cells are what make the latent operator cellwise; the
    preconditioner may be replaced by any lvpp preconditioner -- a direct
    factorization, say -- which keeps the hpG spaces and changes only how the
    Newton systems are solved.

    The public API of an HPG object is:

      HPG(discretization, u, bounds, ...):  build the preset; ``discretization``
            is an HPGDiscretization or a ``(mesh, p)`` pair, ``u`` lives on
            ``discretization.primal``, ``bounds`` is a Constraint or a
            ``(lower, upper)`` pair as for LVPP, and ``bcs`` are the essential
            boundary conditions

      solve():  run the LVPP proximal loop on these spaces; the recorded hpG
            numbers use ``alpha_schedule="double_exponential"``,
            ``alpha_parameters={"alpha_max": 10.0}``, ``increment_norm="H1"``
            and ``solve(tol=1e-4)``

      discretization:  the spaces actually in use, whose ``.primal`` and
            ``.latent`` are reachable without rebuilding them

    A typical call is:

    .. code-block:: python3

      disc = HPGDiscretization.uniform(16, 2)                  # CG_2 x DQ_0 spectral
      u = Function(disc.primal)
      solver = HPG(disc, u, bounds=(lb, None), energy=energy, bcs=bc,
                   alpha_schedule="double_exponential",
                   alpha_parameters={"alpha_max": 10.0},
                   increment_norm="H1")
      solver.solve(tol=1e-4)
      print(solver.u_out[0])                                   # the reconstruction

    Every remaining keyword argument is forwarded to
    :class:`~lvpp.solver.LVPP` verbatim, so every schedule and stopping default
    is LVPP's.
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
