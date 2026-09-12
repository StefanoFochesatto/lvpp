"""The floorless Schur fieldsplit: the scalable production configuration.

Provenance, not taste: every value below is taken verbatim from the archived
experiments, which is what makes this class reproducible rather than plausible.

:data:`SP_WINNER` in ``experiments/archive_2026-09/eps_schedule.py`` is the
winner of the 8-point design ladder in ``RESEARCH.md`` Finding 6.  It
reproduces the direct-LU reference Newton counts and errors *identically* at
every level (8/21, 11/23, 8/19) from an assembled ``Pmat`` with the floor
switched off (``psi_floor = 0``), ``schur``/``upper``, ``use_amat: False`` and
``selfp``: the approximate Schur complement ``Sp = D - B diag(K)^-1 B^T`` is
nonsingular without a floor, because the dead rows of ``D`` are carried by the
coupling term.  This is the "RESOLVED" section of Finding 6: ``psi_floor = 0``
is the recommended setting on this path, and the floor survives as a necessity
only for the additive configuration
(:mod:`lvpp.preconditioners.floor`).

:data:`SP_GAMG_BASE` in ``experiments/archive_2026-09/scale_up.py`` is the same
configuration with ``fieldsplit_0`` (the primal block ``K``, the healthy one)
solved by CG + GAMG instead of MUMPS-LU.  MUMPS is a serial wall, so this is the
scaling variant; ``inner_rtol``/``inner_maxit`` are the run-time knobs the
archive exposed as ``--gamg-rtol``.

Both are options-only: no Python PC, no ``SaddleView``.
"""

from .base import PreconditionerBase

__all__ = ["SchurFieldsplit"]

# eps_schedule.py SP_WINNER, verbatim.
_SCHUR_UPPER_SELFP = {
    # matrix-free action on the operator, assembled preconditioner matrix; the
    # floor/regularization lives on Pmat only, per the Jp-only rule.
    "mat_type": "matfree",
    "pmat_type": "aij",
    "pc_fieldsplit_use_amat": False,
    # outer Newton/Krylov: Newton is exact (LU-exact errors), so the outer solver
    # only has to hit rtol 1e-6 on the Schur-preconditioned system.
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxlambda": 1.0,
    "snes_rtol": 1e-6,
    "snes_max_it": 100,
    "ksp_type": "gmres",
    "ksp_rtol": 1e-6,
    "ksp_max_it": 1000,
    "ksp_gmres_restart": 250,
    "ksp_converged_reason": None,
    # the factorization: upper Schur, approximated by selfp = Sp built from
    # diag(K); both diagonal blocks solved exactly.
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",
    "pc_fieldsplit_schur_precondition": "selfp",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu",
    "fieldsplit_1_pc_factor_mat_solver_type": "mumps",
}

# scale_up.py SP_GAMG_BASE: only fieldsplit_0 changes.  The archived dict also
# carries `fieldsplit_0_pc_factor_mat_solver_type: mumps`, unused under
# `pc_type: gamg`, which is left out here.
_GAMG_FIELDSPLIT_0 = {
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_ksp_converged_reason": None,
    "fieldsplit_0_pc_type": "gamg",
    "fieldsplit_0_pc_gamg_type": "agg",
    "fieldsplit_0_pc_gamg_agg_nsmooths": "1",
    "fieldsplit_0_mg_levels_ksp_type": "chebyshev",
    "fieldsplit_0_mg_levels_pc_type": "jacobi",
}


class SchurFieldsplit(PreconditionerBase):
    """Schur fieldsplit on the mixed saddle system (options-only).

    The preconditioner approximates the exact block factorization of the mixed
    Jacobian by an upper triangular one, whose two diagonal blocks are each
    solved exactly: ``fieldsplit_0`` with MUMPS-LU on the primal block ``K``,
    and ``fieldsplit_1`` with MUMPS-LU on the Schur complement.  The
    approximation sits in that Schur complement.  The true one uses ``K^-1``,
    and PETSc's ``selfp`` option replaces it by a diagonal approximation, giving
    ``Sp = D - B diag(K)^-1 B^T`` at the cost of no assembly beyond ``D`` and
    ``diag(K)``; the coupling term it retains is what keeps the dead rows of
    ``D`` from making the preconditioner singular.

    ``inner_rtol`` is the tolerance for the CG solve of the primal block and
    ``inner_maxit`` its iteration cap; both are used only when ``gamg`` is True,
    since the default LU blocks are exact.  ``gamg`` replaces ``fieldsplit_0``'s
    MUMPS-LU with CG + GAMG (``SP_GAMG_BASE``), leaving ``fieldsplit_1``, the
    Schur settings and the outer solver unchanged.
    """

    def __init__(self, inner_rtol=1e-6, inner_maxit=1000, gamg=False):
        self.inner_rtol = inner_rtol
        self.inner_maxit = inner_maxit
        self.gamg = bool(gamg)

    def parameters(self):
        params = dict(_SCHUR_UPPER_SELFP)
        if not self.gamg:
            return params
        # unused under `fieldsplit_0_pc_type: gamg`; dropped for clarity.
        params.pop("fieldsplit_0_pc_factor_mat_solver_type", None)
        params.update(_GAMG_FIELDSPLIT_0)
        params["fieldsplit_0_ksp_rtol"] = self.inner_rtol
        params["fieldsplit_0_ksp_max_it"] = self.inner_maxit
        return params

    def __repr__(self):
        return (f"SchurFieldsplit(inner_rtol={self.inner_rtol!r}, "
                f"inner_maxit={self.inner_maxit!r}, gamg={self.gamg!r})")