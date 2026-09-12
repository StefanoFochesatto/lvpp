"""The degeneracy floor: an additive term on the latent block of ``Jp``.

Why a floor at all
------------------
The LVPP change of variables ``u = phi + exp(psi)`` makes the operator
structural -- fixed sparsity, no active-set row mutation -- at the price of a
degenerate latent block.  As ``psi -> -infinity`` on the contact set, ``u =
phi`` there, so the psi-row of the residual forces ``exp(psi) = u - phi`` to
zero and the diagonal block ``D = diag(M .* exp(psi))`` of the assembled
Jacobian loses those rows entirely.  The loss is not a small perturbation: the
shift ``M .* exp(psi)`` collapsing drags the measured diagonal of ``D`` down to
``1e-28..1e-80`` at depth, and the count of dead rows grows like the contact
area, ``~ h^-2`` (``RESEARCH.md`` Finding 1).  A preconditioner built on ``D``
therefore sees a singular block, and this class floors it.

The floor adds

    sum_j int eps_j(x) psi_j^trial psi_j^test dx

to the latent diagonal, where the field-proportional scale is

    eps_j(x) = constant + drift * alpha * |lambda_j(x)|,

with ``lambda_j = (psi_prev - psi)/alpha`` the proximal drift, that is the
discrete multiplier (LVPP (2.1), Theorem 4.13 of LVPP), and ``alpha`` the
current proximal parameter.  The two branches of ``eps_j`` do different jobs.
The constant branch floors the whole latent block against exact singularity.
The ``drift`` branch tracks the local contact force: it is large where a
multiplier is active, i.e. where the floor is actually needed, and small in the
inactive region, so the block is not flooded the way a constant-only floor
floods it.  At ``constant = 1e-2`` the floor *dominates* ``D`` at depth
(``RESEARCH.md`` Finding 6, spectral anatomy point 4).

Where the floor goes
--------------------
On ``Jp`` only, through :meth:`jacobian_correction`.  On ``J`` it perturbs the
Newton direction, and the outer proximal point iteration needs exact Newton
steps, so it is measured to fail at every mesh level and every tolerance
(``RESEARCH.md`` Finding 1).  :meth:`operator_correction` exposes that failure
mode as a diagnostic: it is off by default and its only purpose is to keep the
negative result reproducible.

When the floor is not needed
----------------------------
The Schur fieldsplit configuration
(:class:`~lvpp.preconditioners.schur.SchurFieldsplit`) is nonsingular without
any floor, because the dead rows of ``D`` are carried by the coupling term
``B diag(K)^-1 B^T`` of the Schur complement, so its recommended setting is
``constant = 0`` (``RESEARCH.md`` Finding 6, "RESOLVED").  The floor is
load-bearing only for the additive/diagonal-inversion path, and even there the
safe window is a band: ``<= 1e-5`` is indistinguishable from zero, ``1e-4`` the
knife edge, ``>= 1e-3`` fails at depth.
"""

from .base import PreconditionerBase

__all__ = ["DegeneracyFloor"]


class DegeneracyFloor(PreconditionerBase):
    """Additive floor on the latent block; a no-op by default.

    ``constant`` is the constant part of ``eps(x)``, and ``0`` switches it off.
    ``drift`` is the coefficient of the field-proportional part
    ``alpha * |lambda(x)|``, with ``lambda`` the live proximal drift read afresh
    at each assembly, and ``0`` switches it off.  ``on_operator=True`` also adds
    the floor to the true Jacobian ``J``, regularizing the operator rather than
    the preconditioner; that is measured to fail at every level
    (``RESEARCH.md`` Finding 1), so it is a diagnostic and should stay False.
    With both parts disabled the class returns ``None`` from every correction,
    and ``solver_parameters``, empty by default, adds PETSc options on top of
    the LU default.
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
        """Return ``eps(x) * inner(psi_trial, psi_test) * dx`` summed over the
        latent blocks, or ``None`` when the floor is disabled."""
        return self._floor(view)

    def operator_correction(self, view):
        """Return the same form on ``J`` when ``on_operator``, else ``None``.

        This violates the Jp-only rule and is the archived negative result
        rather than a knob: an operator-level floor was measured to fail at
        every mesh level and tolerance (``RESEARCH.md`` Finding 1).
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
        # primal dofs come first and contiguously in the mixed space, so the
        # j-th latent block is split(z)[n + j], with n the number of unknowns.
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