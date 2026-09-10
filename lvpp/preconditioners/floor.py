"""The degeneracy floor: an additive term on the latent block of ``Jp``.

Why a floor at all
------------------
The LVPP change of variables ``u = phi + exp(psi)`` makes the operator
structural (fixed sparsity, no active-set row mutation), at the price of a
degenerate latent block.  On the contact interior ``u = phi``, so the psi-row
forces ``exp(psi) = u - phi -> 0`` and the ``D = diag(exp(psi))`` block of the
assembled Jacobian has *exactly zero* rows there (``RESEARCH.md`` Finding 1;
measured ``|D|`` diagonal down to ``1e-28..1e-80`` at depth, with the dead-row
count scaling like the contact area ``~ h^-2``).

A preconditioner built on ``D`` therefore sees a singular block.  The floor
adds

.. math::  \\sum_j \\int \\varepsilon_j(x)\\,\\psi_j^{\\text{trial}}\\,\\psi_j^{\\text{test}}\\,dx

to the latent diagonal, with the field-proportional scale

.. math::  \\varepsilon_j(x) = \\texttt{constant} + \\texttt{drift}\\;\\alpha\\;|\\lambda_j(x)|,

where ``lambda_j = (psi_prev - psi)/alpha`` is the proximal drift, i.e. the
discrete multiplier, and ``alpha`` the current proximal parameter.  The
``drift`` term thus tracks the local contact force instead of flooding the
whole latent block, which is what the constant-only floor does (at
``constant = 1e-2`` the floor *dominates* ``D`` at depth, ``RESEARCH.md``
Finding 6, spectral anatomy point 4).

Where the floor goes
--------------------
On ``Jp`` only, through :meth:`jacobian_correction`.  Putting it on ``J``
perturbs the Newton direction -- the outer proximal point iteration needs
exact Newton steps -- and is measured to fail at every mesh level, all
tolerances (``RESEARCH.md`` Finding 1).  :meth:`operator_correction` exposes
that failure mode as a *documented legacy diagnostic*: it is off by default
and exists so the negative result stays reproducible.

When the floor is not needed
----------------------------
The fluent ``schur``/``upper``/``selfp`` configuration
(:class:`~lvpp.preconditioners.schur.SchurFieldsplit`) is nonsingular without
any floor -- the dead rows of ``D`` are carried by the coupling term
``B diag(K)^{-1} B^T`` -- so its recommended setting is ``constant = 0``
(``RESEARCH.md`` Finding 6, "RESOLVED").  The floor is load-bearing only for
the additive/diagonal-inversion path, and even there the safe window is a
band: ``<= 1e-5`` is indistinguishable from zero, ``1e-4`` the knife edge,
``>= 1e-3`` fails at depth.
"""

from .base import PreconditionerBase

__all__ = ["DegeneracyFloor"]


class DegeneracyFloor(PreconditionerBase):
    """Additive floor on the latent block; a no-op by default.

    Parameters
    ----------
    constant : float
        Constant part of ``eps(x)``.  ``0`` disables it.
    drift : float
        Coefficient of the field-proportional part ``alpha * |lambda(x)|``,
        with ``lambda`` the proximal drift (live; read fresh at each
        assembly).  ``0`` disables it.
    on_operator : bool
        Legacy diagnostic: also add the floor to the true Jacobian ``J``,
        i.e. regularize the operator rather than the preconditioner.
        Measured to fail at every level (``RESEARCH.md`` Finding 1); keep
        False.
    solver_parameters : dict or None
        Extra PETSc options.  Empty by default, so the LU default stands.
    """

    def __init__(self, constant=0.0, drift=0.0, on_operator=False,
                 solver_parameters=None):
        self.constant = float(constant)
        self.drift = float(drift)
        self.on_operator = bool(on_operator)
        self.solver_parameters = dict(solver_parameters or {})

    def parameters(self):
        return dict(self.solver_parameters)

    def jacobian_correction(self, view):
        """``eps(x) * inner(psi_trial, psi_test) * dx``, summed over the latent
        blocks; ``None`` when the floor is disabled."""
        return self._floor(view)

    def operator_correction(self, view):
        """The same form on ``J`` when ``on_operator``; ``None`` otherwise.

        This violates the Jp-only rule.  It is the archived negative result,
        not a knob: ``psi_floor_operator=True`` was measured to fail at every
        mesh level and tolerance (``RESEARCH.md`` Finding 1).
        """
        if not self.on_operator:
            return None
        return self._floor(view)

    def _floor(self, view):
        if self.constant == 0.0 and self.drift == 0.0:
            return None
        import ufl
        from firedrake import TestFunction, TrialFunction, split

        Z = view.z.function_space()
        zt = TestFunction(Z)
        ztr = TrialFunction(Z)
        # Primal dofs come first and contiguously in the mixed space, so the
        # j-th latent block is split(...)[n + j] with n the number of unknowns.
        n = len(view.primal_spaces)
        floor = 0
        for j, unknown in enumerate(view.constraint_unknowns):
            eps_x = self.constant + self.drift * view.alpha * abs(view.drift[j])
            dxj = view.constraint_measure(unknown)
            floor = floor + eps_x * ufl.inner(
                split(ztr)[n + j], split(zt)[n + j]) * dxj
        return floor

    def __repr__(self):
        return (f"DegeneracyFloor(constant={self.constant!r}, "
                f"drift={self.drift!r}, on_operator={self.on_operator!r})")
